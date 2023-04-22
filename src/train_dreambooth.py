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
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.data
import transformers
import transformers.utils.logging
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version

from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

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
from dataload import PromptDataset, DreamBoothDataset, TransformedDataset, transforms_random_crop
from utils import Timer, Timed, timed_wrapper, _make_cached_caller_targ

IS_DEV = os.environ.get("DEV") == "1"

if is_wandb_available():
    import wandb

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
        text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, epoch,
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
            torch_dtype = weight_dtype,
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
    sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size = args.sample_batch_size)

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
    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder = "scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder = "text_encoder", revision = args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder = "vae", revision = args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder = "unet", revision = args.revision,
    )

    return noise_scheduler, text_encoder_cls, text_encoder, vae, unet


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


def init(args, accelerator, text_encoder, text_encoder_cls, unet):
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

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
                sub_dir = "unet" if type(model) == type(unet) else "text_encoder"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                if type(model) == type(text_encoder):
                    # load transformers style into model
                    load_model = text_encoder_cls.from_pretrained(input_dir, subfolder = "text_encoder")
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


def make_dataloaders(args, tokenizer, with_prior_preservation, img_batch_size, reg_batch_size):
    # col = timed_wrapper()(collate_fn)
    col = collate_fn

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root = args.instance_data_dir,
        instance_prompt = args.instance_prompt,
        tokenizer = tokenizer,
    )

    train_dataloader = torch.utils.data.DataLoader(
        TransformedDataset(train_dataset, transforms_random_crop(args.resolution)),
        batch_size = img_batch_size,
        shuffle = True,
        collate_fn = col,
        num_workers = args.dataloader_num_workers,
    )

    train_reg_dataset = None
    if with_prior_preservation:
        train_reg_dataset = DreamBoothDataset(
            instance_data_root = args.class_data_dir,
            instance_prompt = args.class_prompt,
            tokenizer = tokenizer,
        )

        train_reg_dataloader = torch.utils.data.DataLoader(
            TransformedDataset(train_reg_dataset, transforms_random_crop(args.resolution)),
            batch_size = reg_batch_size,
            shuffle = True,
            collate_fn = col,
            num_workers = args.dataloader_num_workers,
        )
    else:
        train_reg_dataloader = itertools.cycle([None])

    return train_dataloader, train_reg_dataloader


def main(args):
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
    reg_batch_size = (img_batch_size * args.regs_per_image) if with_prior_preservation else 0
    batch_size = img_batch_size + reg_batch_size

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

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

    tokenizer = load_tokenizer(args)
    train_dataloader, train_reg_dataloader = make_dataloaders(args, tokenizer, with_prior_preservation, img_batch_size, reg_batch_size)

    with Timed("Model Load", True):
        noise_scheduler, text_encoder_cls, text_encoder, vae, unet = load_model(args)
        vae.requires_grad_(False)

    init(args, accelerator, text_encoder, text_encoder_cls, unet)

    encode_text_stats = None
    if args.train_text_encoder:
        encode_text = text_encoder
        text_encoder.requires_grad_(False)
    else:
        # cache CLIP encodes when not training text encoder
        encode_text, encode_text_stats = _make_cached_caller_targ(text_encoder)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * batch_size * accelerator.num_processes

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

    make_reg = None
    if args.prior_preservation_mode == "image":
        # make_reg = lambda *args: args[1:]
        pass
    elif args.prior_preservation_mode in ("base_img_otf", "base_img_preinit"):
        if args.prior_preservation_base_otf_half:
            c = lambda md: accelerator.prepare(copy.deepcopy(md).requires_grad_(False).half())
        else:
            c = lambda md: accelerator.prepare(copy.deepcopy(md).requires_grad_(False))

        if not args.train_text_encoder:
            pass

        org_text_enc = c(text_encoder) if args.train_text_encoder else text_encoder
        org_encode_text_fn = _make_cached_caller_targ(org_text_enc)[0]

        org = SimpleNamespace(
            vae = c(vae), unet = c(unet), noise_scheduler = noise_scheduler,
            text_encoder = org_text_enc, encode_text_fn = org_encode_text_fn,
        )

        def _move(device):
            for i in [org.vae, org.unet, org.text_encoder]:
                i.to(device)

        def _add(img_noisy_latents, img_timesteps, img_text_states, res):
            org_noisy, org_timesteps, org_text, org_pred = res
            noisy_latents = torch.cat((img_noisy_latents, org_noisy))
            timesteps = torch.cat((img_timesteps, org_timesteps))
            text_states = torch.cat((img_text_states, org_text))
            return noisy_latents, timesteps, text_states, org_pred

        if args.prior_preservation_mode == "base_img_otf":
            def make_reg(reg_batch, *modelargs):
                # encode reg batch at random timesteps with original untrained copy of encoder
                res = _base_otf_reg(reg_batch, org, weight_dtype, args.offset_noise)
                return _add(*modelargs, res)
        else:
            _base_q = queue.Queue()

            def base_img_preinit_warmup(steps, dataloader, move: bool):
                with Timed("base_img_preinit of %s steps" % steps, True):
                    torch.cuda.empty_cache()

                    if move:
                        _move(accelerator.device)

                    for i in range(steps):
                        reg_batch = next(dataloader)
                        res = _base_otf_reg(reg_batch, org, weight_dtype, args.offset_noise)
                        _base_q.put(res)

                    torch.cuda.empty_cache()

                    if move:
                        _move("cpu")

            def make_reg(reg_batch, *modelargs):
                res = _base_q.get()
                return _add(*modelargs, res)

        if IS_DEV:
            _auto_load_hook(org.unet, accelerator, "ORG_UNET")

    elif args.prior_preservation_mode is not None:
        raise ValueError("unsupported prior_preservation_mode", args.prior_preservation_mode)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Optimizer creation
    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr = args.learning_rate,
        betas = (args.adam_beta1, args.adam_beta2),
        weight_decay = args.adam_weight_decay,
        eps = args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer = optimizer,
        num_warmup_steps = args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps = args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles = args.lr_num_cycles,
        power = args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, train_reg_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, train_reg_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, train_reg_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, train_reg_dataloader, lr_scheduler
        )

    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype = weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype = weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    with Timer("accelerator.init_trackers took {} secs"):
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:  # and not IS_DEV:
            accelerator.init_trackers("dreambooth", config = {
                "args":             vars(args),
                "total_batch_size": total_batch_size,
                "num_processes":    accelerator.num_processes,
                "weight_dtype":     str(weight_dtype),
            })

    logger.info("***** Running training *****")
    logger.info(f"  is_main_process = {accelerator.is_main_process}")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num reg examples = {len(train_reg_dataloader.dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {batch_size} ({img_batch_size} img + {reg_batch_size} reg)")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    resume_step = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key = lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable = not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    if IS_DEV:
        unet.set_attention_slice(1)
        unet.set_attn_processor

        def enc(model, org_fn, *args, **kwargs):
            with Timer("VAE TO Device took {}"):
                model.to(accelerator.device)
            res = org_fn(*args, **kwargs)
            with Timer("VAE TO CPU took {}"):
                model.to("cpu")
            return res

        vae.encode = partial(enc, vae, vae.encode)

    train_reg_dataloader_cycle = itertools.cycle(train_reg_dataloader)

    # if IS_DEV:
    #     log_validation(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, 0)

    MiniTimer = Timer
    MiniTimer = contextlib.nullcontext

    print("starting", os.getpid())
    for epoch in range(first_epoch, args.num_train_epochs):
        epoch_timer = Timed("EPOCH")
        epoch_timer.__enter__()

        unet.train()
        if args.train_text_encoder:
            text_encoder.train()

        if args.prior_preservation_mode == "base_img_preinit":
            steps_needed = len(train_dataloader.dataset) - (resume_step or 0)
            base_img_preinit_warmup(steps_needed, train_reg_dataloader_cycle, True)
            req_iter = itertools.cycle([None])
        else:
            req_iter = train_reg_dataloader_cycle

        for step, (batch, reg_batch) in enumerate(zip(train_dataloader, req_iter)):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert images to latent space
                pixels = batch["pixel_values"]
                if reg_batch and "pixel_values" in reg_batch:
                    pixels = torch.cat((pixels, reg_batch["pixel_values"]))
                with MiniTimer("VAE encode of %s images took {}" % (pixels.shape,)):
                    latents = vae.encode(pixels.to(dtype = weight_dtype)).latent_dist.sample()
                    del pixels
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                if args.offset_noise:
                    noise = torch.randn_like(latents) + 0.1 * torch.randn(
                        latents.shape[0], latents.shape[1], 1, 1, device = latents.device
                    )
                else:
                    noise = torch.randn_like(latents)

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device = latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                input_ids = batch["input_ids"]
                if reg_batch and "input_ids" in reg_batch:
                    input_ids = torch.cat((input_ids, reg_batch["input_ids"]))

                with MiniTimer("CLIP of batch %s took {}" % (input_ids.shape,)):
                    encoder_hidden_states = encode_text(input_ids).last_hidden_state
                    del input_ids

                reg_target = None
                if make_reg is not None:
                    with torch.no_grad():
                        noisy_latents, timesteps, encoder_hidden_states, reg_target = \
                            make_reg(reg_batch, noisy_latents, timesteps, encoder_hidden_states)

                with MiniTimer("Unet of %s latents took {}" % (noisy_latents.shape,)):
                    # Predict the noise residual
                    assert encoder_hidden_states.shape[0] == noisy_latents.shape[0]
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim = 0)
                    if reg_target is not None:
                        target_prior = reg_target
                    else:
                        target, target_prior = torch.chunk(target, 2, dim = 0)

                    # Compute instance loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction = "mean")

                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction = "mean")
                    prior_loss_adj = prior_loss * args.prior_loss_weight
                    print("prior_loss {} * {} = {} ".format(prior_loss.item(), args.prior_loss_weight, prior_loss_adj.item()))

                    # Add the prior loss to the instance loss.
                    loss = loss + prior_loss_adj
                    accelerator.log({
                        "loss":            loss.item(),
                        "prior_loss":      prior_loss.item(),
                        "prior_loss_adj":  prior_loss_adj.item(),
                        "loss_prio_ratio": (loss / prior_loss).item(),
                        "epoch":           epoch,
                    }, step = step)
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction = "mean")
                    accelerator.log({ "loss": loss.item(), "epoch": epoch }, step = step)

                with MiniTimer("backward took {}"):
                    accelerator.backward(loss)

                if accelerator.sync_gradients:
                    with MiniTimer("sync_gradients 1 took {}"):
                        params_to_clip = (
                            itertools.chain(unet.parameters(), text_encoder.parameters())
                            if args.train_text_encoder
                            else unet.parameters()
                        )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                with MiniTimer("Opt Steps took {}"):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none = args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        with Timed("Validation images"):
                            log_validation(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, epoch)

            logs = { "loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0] }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step = global_step)

            if global_step >= args.max_train_steps:
                break

        epoch_timer.__exit__(None, None, None)

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet = accelerator.unwrap_model(unet),
            text_encoder = accelerator.unwrap_model(text_encoder),
            revision = args.revision,
        )
        pipeline.save_pretrained(args.output_dir)
        pipeline.enable_attention_slicing()

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
