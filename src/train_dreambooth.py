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

import os
import time

import torch

IS_DEV = os.environ.get("DEV") == "1"
# if os.getenv("SKIP_IMPORTS") != "1":
import contextlib
import hashlib
import itertools
import logging
import math
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
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
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel, StableDiffusionPipeline

from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from arguments import parse_args
from dataload import PromptDataset, DreamBoothDataset, TransformedDataset, transforms_random_crop, StaticDataset, LatentZipDataset
from utils import Timer, Timed, timed_wrapper, _make_cached_caller_targ

from fastai.callback.core import Callback, CancelBackwardException
from fastai.learner import Learner
from fastai.data.core import DataLoaders

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
diffusers.utils.check_min_version("0.15.0")
print("diffusers", diffusers.__version__)

logger = get_logger(__name__)

try:
    import safetensors

    is_safetensors_available = True
except ImportError:
    is_safetensors_available = False

import sys

if sys.version_info[:2] >= (3, 9):
    from typing import Dict, List, Tuple, Set, Union, Optional, Iterable, Type, Callable, Any, Generator, TypeVar
    from collections.abc import Iterable, Callable, Generator, AsyncGenerator, Collection
else:
    from typing import Dict, List, Tuple, Set, Union, Optional, Iterable, Type, Callable, Any, Generator, TypeVar

# --------------------------------------------------------
# from https://github.com/fastai/course22p2/blob/master/nbs/11_initializing.ipynb
import sys
import traceback
import gc


def clean_ipython_hist():
    # Code in this function mainly copied from IPython source
    if not 'get_ipython' in globals():
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
        text_encoder, tokenizer, unet, vae, args, accelerator, epoch, vae_dtype,
        pipeline = None,
        num_inference_steps = 30,
):
    clean_mem()
    logger.info(
        f"Running validation... \n Generating {args.validation_batches} images with prompt:"
        f" {args.validation_prompt}."
    )

    if IS_DEV:
        num_inference_steps = 15

    if pipeline is None:
        vaeargs = dict(vae = accelerator.unwrap_model(vae)) if vae else { }
        # create pipeline (note: unet and vae are loaded again in float32)
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder = accelerator.unwrap_model(text_encoder),
            tokenizer = tokenizer,
            unet = accelerator.unwrap_model(unet),
            **vaeargs,
            revision = args.revision,
            torch_dtype = vae_dtype,
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

    images = []
    with torch.autocast("cuda"):
        for prompt in args.validation_prompt:
            generator = None if args.seed is None else torch.Generator(device = accelerator.device).manual_seed(args.seed)
            for _ in range(args.validation_batches):
                imgbatch = pipeline(
                    prompt, num_inference_steps = num_inference_steps, generator = generator,
                    num_images_per_prompt = max(1, args.validation_batch_size or 0)
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


def collate_fn(examples: List[dict]) -> dict:
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples if "instance_images" in example] or None
    if pixel_values:
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format = torch.contiguous_format).float()

    latents = [example["latents"] for example in examples if "latents" in example] or None
    if latents:
        latents = torch.cat(latents)
        latents = latents.to(memory_format = torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim = 0)

    batch = {
        "input_ids":    input_ids,
        "pixel_values": pixel_values,
        "latents":      latents,
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
    global_step: Any = None
    global_train_images: Any = None
    global_reg_images: Any = None
    img_batch_size: int = None
    reg_batch_size: int = None
    batch_size: int = None
    dl_need_vae: bool = None
    weight_dtype: Any = None
    learn: Learner = None

    def __post_init__(self):
        self.encode_text_cached, self.encode_text_stats = _make_cached_caller_targ(self.text_encoder)
        self.set_cached(False)

    def set_cached(self, cached = True):
        self.encode_text = self.encode_text_cached if cached else self._encode_text

    def _encode_text(self, ids):
        return self.text_encoder(ids)

    def make_save_path(self, global_step = None):
        global_step = (self.global_step if global_step is None else global_step) or 0
        epoch = self.learn.epoch
        output_dir = Path(self.args.output_dir)
        gstep_t = self.global_train_images or 0
        gstep_r = self.global_reg_images or 0
        model_path = output_dir / f"model_g{global_step}_i{gstep_t + gstep_r}_t{gstep_t}_r{gstep_r}_e{epoch}_b{self.batch_size}_t{int(time.time_ns())}"
        model_path.mkdir(exist_ok = True)
        return str(model_path)


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


def load_model(args, add_vae = True):
    # Load scheduler and models
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder = "text_encoder", revision = args.revision, resume_download = True,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder = "unet", revision = args.revision, resume_download = True,
    )
    vae = None
    if add_vae:
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder = "vae", revision = args.revision, resume_download = True, )
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder = "scheduler")

    return TrainContext(unet, None, text_encoder, vae, noise_scheduler)


def load_dev_model(args, add_vae):
    # make small random compatible models for development

    from transformers import CLIPTextConfig
    unet = UNet2DConditionModel(block_out_channels = (4, 4, 4, 4), cross_attention_dim = 16, norm_num_groups = 4, attention_head_dim = 1)
    tokenizer = load_tokenizer(args)
    text_encoder = CLIPTextModel(CLIPTextConfig(hidden_size = 16, intermediate_size = 16))
    vae = None
    if add_vae:
        vae = AutoencoderKL(block_out_channels = (4,), norm_num_groups = 1)
    noise_scheduler = DDPMScheduler()

    return TrainContext(unet, tokenizer, text_encoder, vae, noise_scheduler)


# def _base_otf_reg(batch, org, weight_dtype, offset_noise, Timer = contextlib.nullcontext):
#     org_dtype = org.vae.dtype
#     pixels = batch["pixel_values"]
#     with Timer("VAE encode of %s images took {}" % (pixels.shape,)):
#         latents = org.vae.encode(pixels.to(dtype = org_dtype)).latent_dist.sample()
#         del pixels
#     latents = latents * org.vae.config.scaling_factor
#
#     if offset_noise:
#         noise = torch.randn_like(latents) + 0.1 * torch.randn(
#             latents.shape[0], latents.shape[1], 1, 1, device = latents.device
#         )
#     else:
#         noise = torch.randn_like(latents)
#
#     timesteps = torch.randint(0, org.noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device = latents.device)
#     timesteps = timesteps.long()
#
#     noisy_latents = org.noise_scheduler.add_noise(latents, noise.to(org_dtype), timesteps)
#
#     input_ids = batch["input_ids"]
#     with Timer("ORG CLIP of batch %s took {}" % (input_ids.shape,)):
#         encoder_hidden_states = org.encode_text_fn(input_ids).last_hidden_state.to(org_dtype)
#         del input_ids
#
#     with Timer("ORG Unet of %s latents took {}" % (latents.shape,)):
#         # Predict the noise residual
#         assert encoder_hidden_states.shape[0] == noisy_latents.shape[0]
#         model_pred = org.unet(noisy_latents, timesteps, encoder_hidden_states).sample
#
#     [i.to(weight_dtype) for i in (noisy_latents, encoder_hidden_states, model_pred)]
#     return noisy_latents, timesteps, encoder_hidden_states, model_pred


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
        print("save state prehook")

        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            assert len(models) == len(weights)
            for model in models:
                if isinstance(model, UNet2DConditionModel):
                    sub_dir = "unet"
                elif isinstance(model, CLIPTextModel):
                    sub_dir = "text_encoder"
                elif isinstance(model, AutoencoderKL):
                    if args.checkpointing_skip_vae:
                        print("skipping saving vae")
                        sub_dir = "vae"
                        weights.pop()
                        continue
                else:
                    raise ValueError("unexpected model type", type(model), model)
                model.save_pretrained(os.path.join(output_dir, sub_dir), safe_serialization = is_safetensors_available)
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

    need_vae = False
    if args.instance_latent_zip:
        assert args.instance_data_dir is None, "both latent_zip and instance_dir"
        train_dataset = LatentZipDataset(
            args.instance_latent_zip,
            instance_prompt = args.instance_prompt,
            tokenizer = tokenizer,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size = img_batch_size,
            shuffle = True,
            collate_fn = col,
            num_workers = args.dataloader_num_workers,
        )
    else:
        need_vae = True
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

    logger.info("setup main data loader with %s images, latent based: %s",
                len(train_dataset), args.instance_latent_zip)

    train_reg_dataloader = None
    if with_prior_preservation:
        if args.class_latent_zip:
            train_reg_dataset = LatentZipDataset(
                args.class_latent_zip,
                instance_prompt = args.class_prompt,
                tokenizer = tokenizer,
            )
            train_reg_dataloader = DataLoader(
                train_reg_dataset,
                batch_size = reg_batch_size,
                shuffle = True,
                collate_fn = col,
                num_workers = args.dataloader_num_workers,
            )

            if len(train_reg_dataset) < args.num_class_images:
                raise ValueError("too few class image latents in zip", args.num_class_images, len(train_reg_dataset))
        else:
            need_vae = True
            train_reg_dataset = DreamBoothDataset(
                instance_data_root = args.class_data_dir,
                instance_prompt = args.class_prompt,
                tokenizer = tokenizer,
            )
            train_reg_dataloader = DataLoader(
                TransformedDataset(train_reg_dataset, transforms_random_crop(args.resolution)),
                batch_size = reg_batch_size,
                shuffle = True,
                collate_fn = col,
                num_workers = args.dataloader_num_workers,
            )

        logger.info("setup reg data loader with %s prior prev images, latent based: %s",
                    len(train_reg_dataset), args.class_latent_zip)

    return train_dataloader, train_reg_dataloader, need_vae


class DevModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)

    def __call__(self, x):
        return self.conv(x)


def _cat(tensors, device = None, dtype = None):
    tensors = list(tensors)
    if device is not None:
        tensors = [i.to(device) for i in tensors]
    if dtype is not None:
        tensors = [i.to(dtype) for i in tensors]

    if len(tensors) == 1:
        return tensors[0]

    return torch.cat(tensors)


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

        self.last_checkpointing_image_steps = None
        self.last_validation_image_steps = None

    def before_fit(self):
        if self.ctx.global_step is None:
            self.ctx.global_step = 0

        if self.ctx.global_train_images is None:
            self.ctx.global_train_images = 0

        if self.ctx.global_reg_images is None:
            self.ctx.global_reg_images = 0

        if self.ctx.accelerator.is_main_process and self.ctx.args.validation_at_start:
            self._validate()

        # if IS_DEV:
        #     save_path = self.ctx.make_save_path()
        #     print("before_fit saving to", save_path)
        #     self.ctx.accelerator.save_state(save_path)
        #     logger.info(f"Saved state to {save_path}")
        #     exit()
        pass

    def _v0(self, batch):
        pixels = batch.get("pixel_values")
        if pixels is not None:
            assert not batch.get("latents")
            latents = self.ctx.vae.encode(pixels.to(dtype = self.ctx.vae.dtype)).latent_dist.sample()
            latents = latents.to(self.ctx.unet.dtype) * self.ctx.vae.config.scaling_factor
            latents.to(self.ctx.accelerator.device)
        else:
            # just hardcode vae scaling, it's not even in the official config and just using the AutoKL default
            # so would have to basically instantiate vae just to get it, not worth
            latents = batch["latents"].to(self.ctx.accelerator.device) * 0.18215

        input_ids = batch["input_ids"]
        return latents, input_ids

    def get_latents(self, batch):
        #
        latents = []
        input_ids = []
        for prefix in ("", "reg_"):
            if batch.get(prefix + "pixel_values") is not None:
                # TODI: encode both together in case of reg images?
                pixels = batch.pop(prefix + "pixel_values")
                latents.append(self.ctx.vae.encode(pixels.to(dtype = self.ctx.vae.dtype)).latent_dist.sample())
            if batch.get(prefix + "latents") is not None:
                latents.append(batch.get(prefix + "latents"))

            if batch.get(prefix + "input_ids") is not None:
                _ids = batch.get(prefix + "input_ids")
                input_ids.append(_ids)
                if prefix == "":
                    self.ctx.global_train_images += _ids.shape[0]
                else:
                    self.ctx.global_reg_images += _ids.shape[0]

        latents = _cat(latents, self.ctx.accelerator.device, self.ctx.unet.dtype)
        input_ids = _cat(input_ids, self.ctx.accelerator.device)
        # just hardcode vae scaling, it's not even in the official config and just using the AutoKL default
        # so would have to basically instantiate vae once at startup (when in latent mode and not using a vae)
        # just to get the value and then delete the vae again, not worth.
        latents = latents * 0.18215
        assert latents.shape[0] == input_ids.shape[0]
        # print("latentsv1", latents.mean(), latents.min(), latents.max(), latents.var(), latents.view(-1)[:10])
        return latents, input_ids

    def before_batch(self):
        self.ctx.global_step += 1
        batch = self.learn.xb[0]
        latents, input_ids = self.get_latents(batch)
        batch["latents"] = latents
        batch["input_ids"] = input_ids

        # Sample noise that we'll add to the latents
        if self.ctx.args.offset_noise:
            noise = batch["noise"] = torch.randn_like(latents) + 0.1 * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1, device = latents.device
            )
        else:
            noise = batch["noise"] = torch.randn_like(latents)

        context = contextlib.nullcontext() if self.ctx.args.train_text_encoder else torch.no_grad()
        with context:
            batch["encoded_text"] = self.ctx.encode_text(input_ids).last_hidden_state

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
        accelerator = self.ctx.accelerator
        with accelerator.accumulate(self.model):
            accelerator.backward(self.learn.loss_grad)

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(self.model.parameters(), self.ctx.args.max_grad_norm)

        # manually call accelerator.backward() here and stop fastai from doing it since
        # it would call loss.backward() not accel.back()
        raise CancelBackwardException()

    def after_batch(self):
        ctx = self.ctx
        args, global_step, accelerator = ctx.args, ctx.global_step, ctx.accelerator

        if accelerator.is_main_process and args.checkpointing_steps:
            save = False
            if args.checkpointing_steps:
                save = global_step % args.checkpointing_steps == 0

            global_images = (self.ctx.global_train_images or 0) + (self.ctx.global_train_images or 0)
            if args.checkpointing_image_steps:
                next = (self.last_checkpointing_image_steps or 0) + args.checkpointing_image_steps
                if global_images >= next:
                    self.last_checkpointing_image_steps = next
                    save = True

            if save:
                save_path = ctx.make_save_path()
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")

            if args.validation_steps and args.validation_prompt is not None and global_step % args.validation_steps == 0:
                self.validate_soon = True

            if args.validation_image_steps and args.validation_prompt is not None:
                next = (self.last_validation_image_steps or 0) + args.validation_image_steps
                if global_images >= next:
                    self.last_validation_image_steps = next
                    self.validate_soon = True

        if self.ctx.accelerator.is_main_process and self.ctx.accelerator.sync_gradients and self.validate_soon:
            self.validate_soon = False
            self._validate()

    def after_fit(self):
        if self.ctx.accelerator.is_main_process and self.ctx.args.validation_at_end:
            self._validate()

    def _validate(self):
        # TODO: only im main process??
        with Timed("Creating Validation images"):
            ctx = self.ctx
            try:
                if ctx.args.validation_clean:
                    self.learn.opt.zero_grad(set_to_none = True)

                log_validation(ctx.text_encoder, ctx.tokenizer, ctx.unet, ctx.vae, ctx.args, ctx.accelerator, self.learn.epoch, ctx.weight_dtype)

                if ctx.args.validation_clean and not ctx.args.set_grads_to_none:
                    self.learn.opt.zero_grad(set_to_none = False)
            except:
                logger.exception("log_validation failed")
            finally:
                clean_mem()

    def after_step(self):
        if self.ctx.args.set_grads_to_none:
            self.learn.opt.zero_grad(set_to_none = True)

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
        # TODO: check that works at the end of one reg iter
        # and doesnt return one smaller batch at last
        assert reg["pixel_values"].shape[0] == self.ctx.reg_batch_size

        is_pixels, is_latents = batch.get("pixel_values") is not None, batch.get("latents") is not None
        assert is_pixels != is_latents
        self.last_img_batchsize = (batch["latents"] if is_latents else batch["pixel_values"]).shape[0]

        # cat reg batch on top of normal batch if exists
        batch["reg_pixel_values"] = reg["pixel_values"]
        batch["reg_input_ids"] = reg["input_ids"]

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
        }, step = self.learn.ctx.global_step)

        return F.mse_loss(pred, yb, reduction = "mean")


class RegBatchLatents(RegBatchImages):
    # Adds N prior preservation images to each batch.
    # Could just do that via main dataloader but since the other prior prev ones make
    # more sense this way just do it for this one via Callback as well.

    name = "PriorRegCb"
    run_valid = False
    learn: Learner
    order = -1

    def before_batch(self):
        batch = self.learn.xb[0]
        reg = next(self.reg_dl_cycle)[0]

        assert reg["latents"].shape[0] == reg["input_ids"].shape[0]
        assert reg["latents"].shape[0] == self.ctx.reg_batch_size

        is_pixels, is_latents = batch.get("pixel_values") is not None, batch.get("latents") is not None
        assert is_pixels != is_latents
        self.last_img_batchsize = (batch["latents"] if is_latents else batch["pixel_values"]).shape[0]

        # cat reg batch on top of normal batch if exists
        batch["reg_latents"] = reg["latents"]
        batch["reg_input_ids"] = reg["input_ids"]


class PrintProgressCB(Callback):
    run_valid = False
    learn: Learner
    order = -10

    def before_epoch(self):
        print(f"starting epoch {self.learn.epoch} of {self.learn.n_epoch}, global step {self.learn.ctx.global_step}")

    def before_batch(self):
        print(f"starting batch {self.learn.iter} of {self.learn.n_iter}, global step {self.learn.ctx.global_step}")


class TqdmProgressCB(Callback):
    run_valid = False
    learn: Learner
    order = -10

    def __init__(self):
        super().__init__()
        self.pbar = None

    def before_epoch(self):
        print(f"starting epoch {self.learn.epoch} of {self.learn.n_epoch}, global step {self.learn.ctx.global_step}")

    def before_batch(self):
        # print(f"starting batch {self.learn.iter} of {self.learn.n_iter}")
        if self.pbar is None:
            self.pbar = tqdm(total = self.learn.n_iter)

    def after_batch(self):
        self.pbar.update(1)

    def after_epoch(self):
        if self.pbar:
            self.pbar.close()


class Model(torch.nn.Module):
    def __init__(self, ctx: TrainContext, add_all = False):
        super().__init__()
        ctx.model = self
        self.ctx = ctx
        self.learn = None

        # assign so they're part of paramters()
        self.unet = ctx.unet
        if ctx.args.train_text_encoder or add_all:
            self.text_encoder = ctx.text_encoder

        if add_all:
            self.vae = ctx.vae

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
        new_lr = args.learning_rate * args.gradient_accumulation_steps * batch_size * accelerator.num_processes
        logger.info("learing_rate scaling: learning_rate %s = learning_rate %s * gradient_accumulation_steps %s * batch_size %s * num_processes %s",
                    new_lr, args.learning_rate, args.gradient_accumulation_steps, batch_size, accelerator.num_processes)
        args.learning_rate = new_lr

    # Generate class images if prior preservation is enabled.
    if args.prior_preservation_mode in { "image" }:
        if not args.class_latent_zip:
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
        tokenizer = load_tokenizer(args)
        train_dataloader, train_reg_dataloader, dl_need_vae = make_dataloaders(args, tokenizer, with_prior_preservation, img_batch_size, reg_batch_size, True)
        print("dl_need_vae", dl_need_vae)
        ctx = load_model(args, dl_need_vae) if not devmodel else load_dev_model(args, dl_need_vae)
        ctx.train_dataloader, ctx.train_reg_dataloader = train_dataloader, train_reg_dataloader
        ctx.tokenizer = load_tokenizer(args)
        ctx.set_cached(not args.train_text_encoder)
        if ctx.vae is not None:
            ctx.vae.requires_grad_(False)
        ctx.text_encoder.requires_grad_(bool(args.train_text_encoder))
        ctx.args = args
        ctx.accelerator = accelerator
        ctx.img_batch_size = img_batch_size
        ctx.reg_batch_size = reg_batch_size
        ctx.batch_size = batch_size
        ctx.dl_need_vae = dl_need_vae

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

    ctx.weight_dtype = weight_dtype

    if ctx.vae:
        ctx.vae = ctx.vae.to(dtype = weight_dtype)
        if ctx.dl_need_vae:
            # only move to GPU if using image based inputs, not needed if latents already
            # TODO: delete if not needed?
            ctx.vae.to(accelerator.device)

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
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch  # TODO: implement or remove
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = ctx.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    callbacks = []
    if args.prior_preservation_mode == "image":
        assert ctx.train_reg_dataloader is not None
        if ctx.args.class_latent_zip:
            callbacks.append(RegBatchLatents(ctx.train_reg_dataloader, ctx, args.prior_loss_weight))
        else:
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

    try:
        import tqdm
        callbacks.append(TqdmProgressCB())
    except:
        callbacks.append(PrintProgressCB())

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
    logger.info("args %s", ctx.args)

    for k, v in sorted(vars(ctx.args).items()):
        print(f"{k:<30s}: {v}")

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

        if ctx.vae:
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


def get_loss_fn(callbacks, default):
    loss_fn = default

    with_loss = [i for i in (callbacks or []) if hasattr(i, "loss")]
    if with_loss:
        assert len(with_loss) == 1, "multiple losses"
        loss_fn = with_loss[0].loss
        print("using cb loss function", loss_fn)

    return loss_fn


def main(args, run_lr_find: bool = False):
    ctx = setup1(args, devmodel = IS_DEV)
    callbacks = setup2(ctx, init_trackers = not run_lr_find)
    model = Model(ctx)

    val_dataset = StaticDataset([(None, None)])  # use the diffusers log_validation() for val
    dls = DataLoaders(ctx.train_dataloader, val_dataset)

    trainercb = Trainer(ctx)
    optim = make_optimizer(ctx.args)
    loss_fn = get_loss_fn(callbacks, trainercb.loss)

    if run_lr_find:  # or IS_DEV:
        print("lr_find")
        import fastai.callback.schedule
        args.validation_steps, args.validation_image_steps = args.checkpointing_steps = args.checkpointing_image_steps = None
        ctx.learn = learn = Learner(dls, model, loss_fn, optim, args.learning_rate, cbs = [trainercb] + callbacks)
        learn.lr_find()
        return learn

    optim = optim(model.parameters(), lr = args.learning_rate)
    ctx.learn = learn = Learner(dls, model, loss_fn, optim, args.learning_rate, cbs = [trainercb] + callbacks)
    ctx.unet.train()
    if args.train_text_encoder:
        ctx.text_encoder.train()

    try:
        learn.fit(args.num_train_epochs)
    except KeyboardInterrupt:
        print("Ctrl-C, fit cancelled")
    except:
        logger.exception("fit failed")

    print("models.encode_text_stats", ctx.encode_text_stats)
    accelerator = ctx.accelerator
    accelerator.log({ "encode_text_stats": ctx.encode_text_stats })

    # Create the pipeline using using the trained modules and save it.
    print("done training, wait for everyone", accelerator.is_main_process)
    accelerator.wait_for_everyone()
    print("done", accelerator.is_main_process)
    if accelerator.is_main_process and not IS_DEV:
        save_path = ctx.make_save_path()
        print("done, saving to", repr(save_path))
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet = accelerator.unwrap_model(ctx.unet),
            text_encoder = accelerator.unwrap_model(ctx.text_encoder),
            revision = args.revision,
            safety_checker = None,
            feature_extractor = None,
            requires_safety_checker = False
        )
        pipeline.save_pretrained(save_path, safe_serialization = is_safetensors_available)
        pipeline.enable_attention_slicing()

    accelerator.end_training()

    # logs = { "loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0] }
    # progress_bar.set_postfix(**logs)
    # accelerator.log(logs, step = global_step)
    print("done", accelerator.is_main_process)
    return


if __name__ == "__main__":
    main(parse_args())
