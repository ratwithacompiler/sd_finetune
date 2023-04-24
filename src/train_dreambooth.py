#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

"""
Features:




"""
import contextlib
import copy
import hashlib
import itertools
import logging
import math
import os
import queue
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import accelerate
import numpy as np
# import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.data
import transformers
import transformers.utils.logging
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from fastai.optimizer import OptimWrapper
from packaging import version
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel, StableDiffusionPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from arguments import parse_args
from dataload import PromptDataset, DreamBoothDataset, TransformedDataset, transforms_random_crop, StaticDataset
from utils import Timer, Timed, timed_wrapper, _make_cached_caller_targ

from fastai.callback.core import Callback
from fastai.learner import Learner, Recorder
from fastai.data.core import DataLoaders

IS_DEV = os.environ.get("DEV") == "1"

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0")

logger = get_logger(__name__)

# --------------------------------------------------------
# from https://github.com/fastai/course22p2/blob/master/nbs/11_initializing.ipynb
import sys
import traceback
import gc


def clean_ipython_hist():
    # Code in this function mainly copied from IPython source
    if not 'get_ipython' in globals():
        print("nope")
        return

    ip = get_ipython()
    user_ns = ip.user_ns
    ip.displayhook.flush()
    pc = ip.displayhook.prompt_count + 1
    for n in range(1, pc):
        user_ns.pop('_i' + repr(n), None)

    user_ns.update(dict(_i = '', _ii = '', _iii = ''))
    hm = ip.history_manager
    hm.input_hist_parsed[:] = [''] * pc
    hm.input_hist_raw[:] = [''] * pc
    hm._i = hm._ii = hm._iii = hm._i00 = ''


def clean_tb():
    # h/t Piotr Czapla
    if hasattr(sys, 'last_traceback'):
        traceback.clear_frames(sys.last_traceback)
        delattr(sys, 'last_traceback')
    if hasattr(sys, 'last_type'):
        delattr(sys, 'last_type')
    if hasattr(sys, 'last_value'):
        delattr(sys, 'last_value')


def clean_mem():
    clean_tb()
    clean_ipython_hist()
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------

@torch.no_grad()
def log_validation(
        text_encoder, tokenizer, unet, vae, args, accelerator, epoch,
        pipeline = None,
        num_inference_steps = 22,
):
    clean_mem()
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )

    if IS_DEV:
        num_inference_steps = 15

    if pipeline is None:
        # create pipeline (note: unet and vae are loaded again in float32)
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder = accelerator.unwrap_model(text_encoder),
            tokenizer = tokenizer,
            unet = accelerator.unwrap_model(unet),
            vae = accelerator.unwrap_model(vae),
            revision = args.revision,
            torch_dtype = vae.dtype,
            safety_checker = None,
            feature_extractor = None,
            requires_safety_checker = False
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable = True)
        # def decode_latents(latents):
        #     latents = latents.to(vae.dtype)
        #     res = StableDiffusionPipeline.decode_latents(pipeline, latents)
        #     res.to(pipeline.unet.dtype)
        #     return res
        #
        # pipeline.decode_latents = decode_latents
        pass

    # run inference
    generator = None if args.seed is None else torch.Generator(device = accelerator.device).manual_seed(args.seed)
    images = []
    for _ in range(args.num_validation_images):
        with torch.autocast("cuda"):
            for prompt in args.validation_prompt:
                imgbatch = pipeline(
                    prompt, num_inference_steps = num_inference_steps, generator = generator,
                    num_images_per_prompt = max(1, args.validation_batch_size)
                ).images
                images.extend(((prompt, i) for i in imgbatch))

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for (_prompt, img) in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats = "NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption = f"{i}: {prompt}") for i, (prompt, image) in enumerate(images)
                    ]
                }
            )

    clean_mem()


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder = "text_encoder",
        revision = revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def collate_fn(examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format = torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim = 0)

    batch = {
        "input_ids":    input_ids,
        "pixel_values": pixel_values,
    }
    return batch


def create_class_images(args, accelerator, num_new_images, class_images_dir: Path, num_cur_images: int):
    torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
    if args.prior_generation_precision == "fp32":
        torch_dtype = torch.float32
    elif args.prior_generation_precision == "fp16":
        torch_dtype = torch.float16
    elif args.prior_generation_precision == "bf16":
        torch_dtype = torch.bfloat16
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype = torch_dtype,
        safety_checker = None,
        revision = args.revision,
    )
    pipeline.set_progress_bar_config(disable = True)

    logger.info(f"Number of class images to sample: {num_new_images}.")

    sample_dataset = PromptDataset(args.class_prompt, num_new_images)
    sample_dataloader = DataLoader(sample_dataset, batch_size = args.sample_batch_size)

    sample_dataloader = accelerator.prepare(sample_dataloader)
    pipeline.to(accelerator.device)

    for example in tqdm(sample_dataloader, desc = "Generating class images", disable = not accelerator.is_local_main_process):
        images = pipeline(example["prompt"]).images

        for i, image in enumerate(images):
            hash_image = hashlib.sha1(image.tobytes()).hexdigest()
            image_filename = class_images_dir / f"{example['index'][i] + num_cur_images}-{hash_image}.jpg"
            image.save(image_filename)

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class TrainContext():
    unet: UNet2DConditionModel
    tokenizer: AutoTokenizer
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    noise_scheduler: DDPMScheduler
    train_dataload: Any = None
    reg_dataload: Any = None
    model: "Model" = None
    args: any = None
    accelerator: Any = None
    step: Any = None
    img_batch_size: int = None
    reg_batch_size: int = None
    batch_size: int = None

    def __post_init__(self):
        self.encode_text_cached, self.encode_text_stats = _make_cached_caller_targ(self.text_encoder)
        self.set_cached(False)

    def set_cached(self, cached = True):
        self.encode_text = self.encode_text_cached if cached else self._encode_text

    def _encode_text(self, ids):
        return self.text_encoder(ids)


def load_tokenizer(args):
    if args.tokenizer_name:
        return AutoTokenizer.from_pretrained(args.tokenizer_name, revision = args.revision, use_fast = False)

    if args.pretrained_model_name_or_path:
        return AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder = "tokenizer",
            revision = args.revision,
            use_fast = False,
        )


def load_model(args):
    tokenizer = load_tokenizer(args)
    # Load scheduler and models
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder = "text_encoder", revision = args.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder = "unet", revision = args.revision,
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder = "vae", revision = args.revision)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder = "scheduler")

    return TrainContext(unet, tokenizer, text_encoder, vae, noise_scheduler)


def load_dev_model(args):
    # make small random compatible models for development

    from transformers import CLIPTextConfig
    unet = UNet2DConditionModel(block_out_channels = (4, 4, 4, 4), cross_attention_dim = 16, norm_num_groups = 4, attention_head_dim = 1)
    tokenizer = load_tokenizer(args)
    text_encoder = CLIPTextModel(CLIPTextConfig(hidden_size = 16, intermediate_size = 16))
    vae = AutoencoderKL(block_out_channels = (4,), norm_num_groups = 1)
    noise_scheduler = DDPMScheduler()

    return TrainContext(unet, tokenizer, text_encoder, vae, noise_scheduler)


def _base_otf_reg(batch, org, weight_dtype, offset_noise, Timer = contextlib.nullcontext):
    org_dtype = org.vae.dtype
    pixels = batch["pixel_values"]
    with Timer("VAE encode of %s images took {}" % (pixels.shape,)):
        latents = org.vae.encode(pixels.to(dtype = org_dtype)).latent_dist.sample()
        del pixels
    latents = latents * org.vae.config.scaling_factor

    if offset_noise:
        noise = torch.randn_like(latents) + 0.1 * torch.randn(
            latents.shape[0], latents.shape[1], 1, 1, device = latents.device
        )
    else:
        noise = torch.randn_like(latents)

    timesteps = torch.randint(0, org.noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device = latents.device)
    timesteps = timesteps.long()

    noisy_latents = org.noise_scheduler.add_noise(latents, noise.to(org_dtype), timesteps)

    input_ids = batch["input_ids"]
    with Timer("ORG CLIP of batch %s took {}" % (input_ids.shape,)):
        encoder_hidden_states = org.encode_text_fn(input_ids).last_hidden_state.to(org_dtype)
        del input_ids

    with Timer("ORG Unet of %s latents took {}" % (latents.shape,)):
        # Predict the noise residual
        assert encoder_hidden_states.shape[0] == noisy_latents.shape[0]
        model_pred = org.unet(noisy_latents, timesteps, encoder_hidden_states).sample

    [i.to(weight_dtype) for i in (noisy_latents, encoder_hidden_states, model_pred)]
    return noisy_latents, timesteps, encoder_hidden_states, model_pred


def _auto_load_hook(model, accelerator, label):
    def _pre(model, *args):
        with Timer("%s TO Device took {}" % (label,)):
            model.to(accelerator.device)

    def _fwd(model, *args):
        with Timer("%s TO CPU took {}" % (label,)):
            model.to("cpu")

    model.register_forward_pre_hook(_pre)
    model.register_forward_hook(_fwd)


def init(ctx: TrainContext):
    args, accelerator = ctx.args, ctx.accelerator
    unet, text_encoder = ctx.unet, ctx.text_encoder

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb as _wan
        global wandb
        wandb = _wan

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    logger.info(accelerator.state, main_process_only = False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            for model in models:
                sub_dir = "unet" if type(model) == type(models.unet) else "text_encoder"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                if type(model) == type(models.text_encoder):
                    # load transformers style into model
                    load_model = type(model).from_pretrained(input_dir, subfolder = "text_encoder")
                    model.config = load_model.config
                else:
                    # load diffusers style into model
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder = "unet")
                    model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok = True)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    if args.train_text_encoder and accelerator.unwrap_model(text_encoder).dtype != torch.float32:
        raise ValueError(
            f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
            f" {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True


def make_dataloaders(args, tokenizer, with_prior_preservation, img_batch_size, reg_batch_size, with_dep = False):
    # col = timed_wrapper()(collate_fn)

    if with_dep:
        col = lambda *args, **kwargs: (collate_fn(*args, **kwargs), None)
    else:
        col = collate_fn

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root = args.instance_data_dir,
        instance_prompt = args.instance_prompt,
        tokenizer = tokenizer,
    )
    train_dataloader = DataLoader(
        TransformedDataset(train_dataset, transforms_random_crop(args.resolution)),
        batch_size = img_batch_size,
        shuffle = True,
        collate_fn = col,
        num_workers = args.dataloader_num_workers,
    )

    train_reg_dataloader = None
    if with_prior_preservation:
        train_reg_dataset = DreamBoothDataset(
            instance_data_root = args.class_data_dir,
            instance_prompt = args.class_prompt,
            tokenizer = tokenizer,
        )
        print("reg_batch_size", reg_batch_size)
        train_reg_dataloader = DataLoader(
            TransformedDataset(train_reg_dataset, transforms_random_crop(args.resolution)),
            batch_size = reg_batch_size,
            shuffle = True,
            collate_fn = col,
            num_workers = args.dataloader_num_workers,
        )

    return train_dataloader, train_reg_dataloader


class DevModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)

    def __call__(self, x):
        return self.conv(x)


class Trainer(Callback):
    name = "TrainCb"
    run_valid = False
    order = 0
    learn: Learner

    def __init__(self, ctx: TrainContext):
        super().__init__()
        self.backward_queue = []
        self.ctx = ctx
        self.validate_soon = False

    def before_fit(self):
        if self.ctx.step is None:
            self.ctx.step = 0

    def before_batch(self):
        self.ctx.step += 1
        batch = self.learn.xb[0]

        pixels = batch.pop("pixel_values")
        latents = self.ctx.vae.encode(pixels.to(dtype = self.ctx.vae.dtype)).latent_dist.sample()
        latents = latents.to(self.ctx.unet.dtype) * self.ctx.vae.config.scaling_factor
        batch["latents"] = latents.to(self.ctx.accelerator.device)

        # Sample noise that we'll add to the latents
        if self.ctx.args.offset_noise:
            noise = batch["noise"] = torch.randn_like(latents) + 0.1 * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1, device = latents.device
            )
        else:
            noise = batch["noise"] = torch.randn_like(latents)
        batch["encoded_text"] = self.ctx.encode_text(batch["input_ids"]).last_hidden_state

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.ctx.noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device = latents.device)
        batch["timesteps"] = timesteps.long()

        noise_scheduler = self.ctx.noise_scheduler
        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        self.learn.yb = (target,)

    def before_backward(self):
        # Need to patch backward to call  self.accelerator.backward() for gradient syncing.
        # Track if the patched backward was called to prevent bugs.
        assert not self.backward_queue
        obj = object()

        accelerator = self.ctx.accelerator

        def bw():
            self.backward_queue.remove(obj)
            with accelerator.accumulate(self.model):
                accelerator.backward(self.learn.loss_grad)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(self.model.parameters(), self.ctx.args.max_grad_norm)

        self.backward_queue.append(obj)
        self.learn.loss_grad.backward = bw

    def before_step(self):
        # make sure patched bw was called, just to prevent bugs
        assert not self.backward_queue

    def after_batch(self):
        ctx = self.ctx
        args, step, accelerator = ctx.args, ctx.step, ctx.accelerator

        if accelerator.is_main_process:
            if step % args.checkpointing_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")

            if args.validation_prompt is not None and step % args.validation_steps == 0:
                self.validate_soon = True

    def after_backward(self):
        ctx = self.ctx
        args, accelerator = ctx.args, ctx.accelerator
        if accelerator.sync_gradients and self.validate_soon:
            with Timed("Validation images"):
                if ctx.args.validation_clean:
                    self.learn.opt.zero_grad(set_to_none = True)

                log_validation(ctx.text_encoder, ctx.tokenizer, ctx.unet, ctx.vae, args, accelerator, self.learn.epoch)

                if ctx.args.validation_clean and not ctx.args.set_grads_to_none:
                    self.learn.opt.zero_grad(set_to_none = False)

    def loss(self, pred, yb):
        if pred is None:
            # validation loss, skipped since using log_validate from diffusers code
            assert not self.learn.training
            return torch.tensor([1337.0])

        return F.mse_loss(pred, yb, reduction = "mean")


class RegBatchImages(Callback):
    # Adds N prior preservation images to each batch.
    # Could just do that via main dataloader but since the other prior prev ones make
    # more sense this way just do it for this one via Callback as well.

    name = "PriorRegCb"
    run_valid = False
    learn: Learner
    order = -1

    def __init__(self, reg_dl: DataLoader, ctx: TrainContext, prior_loss_weight: float):
        super().__init__()
        self.reg_dl = reg_dl
        self.ctx = ctx
        self.reg_dl_cycle = itertools.cycle(self.reg_dl)
        self.prior_loss_weight = prior_loss_weight

        self.last_img_batchsize = None

    def before_batch(self):
        batch = self.learn.xb[0]
        reg = next(self.reg_dl_cycle)[0]

        assert reg["pixel_values"].shape[0] == reg["input_ids"].shape[0]
        assert reg["pixel_values"].shape[0] == self.ctx.reg_batch_size

        self.last_img_batchsize = batch["pixel_values"].shape[0]

        # cat reg batch on top of normal batch
        batch["pixel_values"] = torch.cat((batch["pixel_values"], reg["pixel_values"].to(batch["pixel_values"].device)))
        batch["input_ids"] = torch.cat((batch["input_ids"], reg["input_ids"].to(batch["input_ids"].device)))

    def loss(self, pred, yb):
        if pred is None:
            # validation loss, skipped since using log_validate from diffusers code
            assert not self.learn.training
            return torch.tensor([1337.0])

        reg_batch_size = self.ctx.reg_batch_size

        # index from end since last batch per epoch might have less than img_batch_size images
        model_pred = pred[:-reg_batch_size]
        model_pred_prior = pred[-reg_batch_size:]
        target = yb[:-reg_batch_size]
        target_prior = yb[-reg_batch_size:]

        assert model_pred.shape[0] == self.last_img_batchsize
        assert model_pred_prior.shape[0] == reg_batch_size
        assert target.shape[0] == self.last_img_batchsize
        assert target_prior.shape[0] == reg_batch_size

        # Compute instance loss
        loss = F.mse_loss(model_pred.float(), target.float(), reduction = "mean")

        # Compute prior loss
        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction = "mean")
        prior_loss_adj = prior_loss * self.prior_loss_weight
        # print("prior_loss {} * {} = {} ".format(prior_loss.item(), self.prior_loss_weight, prior_loss_adj.item()))

        # Add the prior loss to the instance loss.
        loss = loss + prior_loss_adj
        self.learn.ctx.accelerator.log({
            "loss":            loss.item(),
            "prior_loss":      prior_loss.item(),
            "prior_loss_adj":  prior_loss_adj.item(),
            "loss_prio_ratio": (loss / prior_loss).item(),
        }, step = self.learn.ctx.step)

        return F.mse_loss(pred, yb, reduction = "mean")


class Model(torch.nn.Module):
    def __init__(self, ctx: TrainContext):
        super().__init__()
        ctx.model = self
        self.ctx = ctx
        self.learn = None

        # assign so they're part of paramters()
        self.unet = ctx.unet
        if ctx.args.train_text_encoder:
            self.text_encoder = ctx.text_encoder

    def __call__(self, batch):
        if not self.training and batch is None:
            return None

        c = self.ctx
        vae, noise_scheduler = c.vae, c.noise_scheduler

        latents = batch["latents"]
        noise = batch["noise"]
        timesteps = batch["timesteps"]
        encoder_hidden_states = batch["encoded_text"]

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        # Predict the noise residual
        assert encoder_hidden_states.shape[0] == noisy_latents.shape[0]
        model_pred = c.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        return model_pred


def setup1(args, devmodel = False):
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit = args.checkpoints_total_limit)
    accelerator = Accelerator(
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        mixed_precision = args.mixed_precision,
        log_with = args.report_to,
        logging_dir = logging_dir,
        project_config = accelerator_project_config,
        cpu = args.cpu,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt = "%m/%d/%Y %H:%M:%S",
        level = logging.INFO,
    )

    with_prior_preservation = args.prior_preservation_mode
    img_batch_size = args.train_batch_size
    reg_batch_size = args.reg_batch_size if with_prior_preservation else 0
    batch_size = img_batch_size + reg_batch_size

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * batch_size * accelerator.num_processes

    # Generate class images if prior preservation is enabled.
    if args.prior_preservation_mode in { "image" }:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            raise ValueError("class data dir not found", class_images_dir)

        cur_class_images = len(list(class_images_dir.iterdir()))
        if cur_class_images < args.num_class_images:
            with Timed("Class Image Creation", True):
                num_new_images = args.num_class_images - cur_class_images
                create_class_images(args, accelerator, num_new_images, class_images_dir, cur_class_images)
                raise ValueError()

    with Timed("Model Load", True):
        ctx = load_model(args) if not devmodel else load_dev_model(args)
        ctx.set_cached(not args.train_text_encoder)
        ctx.vae.requires_grad_(False)
        ctx.text_encoder.requires_grad_(bool(args.train_text_encoder))
        ctx.args = args
        ctx.accelerator = accelerator
        ctx.train_dataloader, ctx.train_reg_dataloader = make_dataloaders(args, ctx.tokenizer, with_prior_preservation, img_batch_size, reg_batch_size, True)
        ctx.img_batch_size = img_batch_size
        ctx.reg_batch_size = reg_batch_size
        ctx.batch_size = batch_size

    init(ctx)
    return ctx


def setup2(ctx: TrainContext, init_trackers = True):
    accelerator, args = ctx.accelerator, ctx.args
    # Prepare everything with our `accelerator`.
    ctx.unet, ctx.vae = accelerator.prepare(ctx.unet, ctx.vae)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype
    ctx.vae.to(accelerator.device, dtype = weight_dtype)
    if not args.train_text_encoder:
        ctx.text_encoder.to(accelerator.device, dtype = weight_dtype)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(ctx.train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    ctx.train_dataloader, ctx.train_reg_dataloader = accelerator.prepare(ctx.train_dataloader, ctx.train_reg_dataloader)
    if args.train_text_encoder:
        ctx.text_encoder = accelerator.prepare(ctx.text_encoder)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(ctx.train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = ctx.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    callbacks = []
    if args.prior_preservation_mode == "image":
        assert ctx.train_reg_dataloader is not None
        callbacks.append(RegBatchImages(ctx.train_reg_dataloader, ctx, args.prior_loss_weight))

    # if args.prior_preservation_mode == "base_img_otf":
    #     callbacks.append(RegBatchImages(ctx.train_reg_dataloader, img_batch_size, reg_batch_size, args.prior_loss_weight))
    # elif args.prior_preservation_mode in []:  # ("base_img_otf", "base_img_preinit"):
    #     if args.prior_preservation_base_otf_half:
    #         c = lambda md: accelerator.prepare(copy.deepcopy(md).requires_grad_(False).half())
    #     else:
    #         c = lambda md: accelerator.prepare(copy.deepcopy(md).requires_grad_(False))
    #
    #     if not args.train_text_encoder:
    #         pass
    #
    #     org_text_enc = c(text_encoder) if args.train_text_encoder else text_encoder
    #     org_encode_text_fn = _make_cached_caller_targ(org_text_enc)[0]
    #
    #     org = SimpleNamespace(
    #         vae = c(vae), unet = c(unet), noise_scheduler = noise_scheduler,
    #         text_encoder = org_text_enc, encode_text_fn = org_encode_text_fn,
    #     )
    #
    #     def _move(device):
    #         for i in [org.vae, org.unet, org.text_encoder]:
    #             i.to(device)
    #
    #     def _add(img_noisy_latents, img_timesteps, img_text_states, res):
    #         org_noisy, org_timesteps, org_text, org_pred = res
    #         noisy_latents = torch.cat((img_noisy_latents, org_noisy))
    #         timesteps = torch.cat((img_timesteps, org_timesteps))
    #         text_states = torch.cat((img_text_states, org_text))
    #         return noisy_latents, timesteps, text_states, org_pred
    #
    #     if args.prior_preservation_mode == "base_img_otf":
    #         def make_reg(reg_batch, *modelargs):
    #             # encode reg batch at random timesteps with original untrained copy of encoder
    #             res = _base_otf_reg(reg_batch, org, weight_dtype, args.offset_noise)
    #             return _add(*modelargs, res)
    #     else:
    #         _base_q = queue.Queue()
    #
    #         def base_img_preinit_warmup(steps, dataloader, move: bool):
    #             with Timed("base_img_preinit of %s steps" % steps, True):
    #                 torch.cuda.empty_cache()
    #
    #                 if move:
    #                     _move(accelerator.device)
    #
    #                 for i in range(steps):
    #                     reg_batch = next(dataloader)
    #                     res = _base_otf_reg(reg_batch, org, weight_dtype, args.offset_noise)
    #                     _base_q.put(res)
    #
    #                 torch.cuda.empty_cache()
    #
    #                 if move:
    #                     _move("cpu")
    #
    #         def make_reg(reg_batch, *modelargs):
    #             res = _base_q.get()
    #             return _add(*modelargs, res)
    #
    #     if IS_DEV:
    #         _auto_load_hook(org.unet, accelerator, "ORG_UNET")

    elif args.prior_preservation_mode is not None:
        raise ValueError("unsupported prior_preservation_mode", args.prior_preservation_mode)

    with Timer("accelerator.init_trackers took {} secs"):
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:  # and not IS_DEV:
            if init_trackers:
                accelerator.init_trackers("dreambooth", config = {
                    "args":             vars(args),
                    "total_batch_size": total_batch_size,
                    "num_processes":    accelerator.num_processes,
                    "weight_dtype":     str(weight_dtype),
                })
            else:
                if not hasattr(accelerator, "trackers"):
                    accelerator.trackers = []
    logger.info("***** Running training *****")
    logger.info(f"  is_main_process = {accelerator.is_main_process}")
    logger.info(f"  Num examples = {len(ctx.train_dataloader.dataset)}")
    logger.info(f"  Num reg examples = {len(ctx.train_reg_dataloader.dataset) if ctx.train_reg_dataloader else None}")
    logger.info(f"  Num batches each epoch = {len(ctx.train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {ctx.batch_size} ({ctx.img_batch_size} img + {ctx.reg_batch_size} reg)")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    if IS_DEV:
        ctx.unet.set_attention_slice(1)

        # ctx.unet.set_attn_processor

        def enc(model, org_fn, *args, **kwargs):
            with Timer("VAE TO Device took {}"):
                model.to(accelerator.device)
            res = org_fn(*args, **kwargs)
            with Timer("VAE TO CPU took {}"):
                model.to("cpu")
            return res

        ctx.vae.encode = partial(enc, ctx.vae, ctx.vae.encode)

    # train_reg_dataloader_cycle = itertools.cycle(train_reg_dataloader)
    # if IS_DEV:
    #     log_validation(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, 0)

    return callbacks


def make_optimizer(args):
    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    if IS_DEV:
        optimizer_class = lambda *args, betas = None, weight_decay = None, eps = None, **kwargs: torch.optim.SGD(*args, **kwargs)

    opt_args = partial(
        optimizer_class,
        betas = (args.adam_beta1, args.adam_beta2),
        weight_decay = args.adam_weight_decay,
        eps = args.adam_epsilon,
    )
    return partial(OptimWrapper, opt = opt_args)


def main(args):
    ctx = setup1(args, devmodel = IS_DEV)
    callbacks = setup2(ctx)
    model = Model(ctx)

    val_dataset = StaticDataset([(None, None)])
    dls = DataLoaders(ctx.train_dataloader, val_dataset)

    trainercb = Trainer(ctx)
    optim = make_optimizer(ctx.args)

    loss_fn = trainercb.loss
    with_loss = [i for i in (callbacks or []) if hasattr(i, "loss")]
    if with_loss:
        assert len(with_loss) == 1
        loss_fn = with_loss[0].loss
        print("using cb loss function", loss_fn)

    # from fastai.callback.schedule import lr_find
    # from fastai.learner import Recorder
    # rec = Recorder()
    # learn = Learner(dls, model, loss_fn, optim, args.learning_rate, cbs = [trainercb, rec] + callbacks)
    # learn.lr_find()
    # return

    learn = Learner(dls, model, loss_fn, optim, args.learning_rate, cbs = [trainercb] + callbacks)
    ctx.unet.train()
    if args.train_text_encoder:
        ctx.text_encoder.train()
    try:
        learn.fit(1)
    except KeyboardInterrupt:
        print("Ctrl-C, fit cancelled")
    except:
        logger.exception("fit failed")

    print("models.encode_text_stats", ctx.encode_text_stats)
    accelerator = ctx.accelerator
    accelerator.log({ "encode_text_stats": ctx.encode_text_stats })

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and not IS_DEV:
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet = accelerator.unwrap_model(ctx.unet),
            text_encoder = accelerator.unwrap_model(ctx.text_encoder),
            revision = args.revision,
            safety_checker = None,
            feature_extractor = None,
            requires_safety_checker = False
        )
        pipeline.save_pretrained(args.output_dir)
        pipeline.enable_attention_slicing()

    accelerator.end_training()

    # logs = { "loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0] }
    # progress_bar.set_postfix(**logs)
    # accelerator.log(logs, step = global_step)
    return


if __name__ == "__main__":
    main(parse_args())
