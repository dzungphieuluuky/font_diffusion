"""
Optimized model building for FontDiffuser - BACKWARD COMPATIBLE version
Maintains exact architecture to load pretrained weights
"""

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from functools import lru_cache
from src import ContentEncoder, StyleEncoder, UNet, SCR


def build_unet_optimized(args, optimize=True):
    """
    Build UNet with inference optimizations that DON'T change architecture

    Args:
        args: Configuration arguments
        optimize: Enable inference optimizations (memory format, attention)
    """
    # IMPORTANT: Keep EXACTLY the same architecture as original build.py
    unet = UNet(
        sample_size=args.resolution,
        in_channels=3,
        out_channels=3,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=(
            "DownBlock2D",
            "MCADownBlock2D",
            "MCADownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "StyleRSIUpBlock2D",
            "StyleRSIUpBlock2D",
            "UpBlock2D",
        ),
        block_out_channels=args.unet_channels,
        layers_per_block=2,  # DO NOT CHANGE - must match pretrained
        downsample_padding=1,
        mid_block_scale_factor=1,
        act_fn="silu",
        norm_num_groups=32,  # DO NOT CHANGE - must match pretrained
        norm_eps=1e-05,
        cross_attention_dim=args.style_start_channel * 16,
        attention_head_dim=1,
        channel_attn=args.channel_attn,
        content_encoder_downsample_size=args.content_encoder_downsample_size,
        content_start_channel=args.content_start_channel,
        reduction=32,  # DO NOT CHANGE - must match pretrained
    )

    # Apply only NON-ARCHITECTURAL optimizations
    if optimize:
        # Use channels last memory format (memory optimization only)
        unet = unet.to(memory_format=torch.channels_last)

        # Use memory efficient attention (PyTorch 2.0 optimization)
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            try:
                unet.set_attn_processor(AttnProcessor2_0())
                print("✓ Using memory efficient attention (AttnProcessor2_0)")
            except:
                print("⚠ Could not set AttnProcessor2_0, using default")

    return unet


@lru_cache(maxsize=1)
def build_unet_cached(args, optimize=True):
    """
    Cached version of build_unet for repeated inference sessions
    """
    return build_unet_optimized(args, optimize)


def build_style_encoder_optimized(args, optimize=True):
    """
    Build style encoder with memory optimizations only
    """
    style_image_encoder = StyleEncoder(
        G_ch=args.style_start_channel, resolution=args.style_image_size[0]
    )

    if optimize:
        # Channels last format (memory optimization only)
        style_image_encoder = style_image_encoder.to(memory_format=torch.channels_last)

    print("Get CG-GAN Style Encoder!")
    return style_image_encoder


@lru_cache(maxsize=1)
def build_style_encoder_cached(args, optimize=True):
    """
    Cached version of style encoder
    """
    return build_style_encoder_optimized(args, optimize)


def build_content_encoder_optimized(args, optimize=True):
    """
    Build content encoder with memory optimizations only
    """
    content_image_encoder = ContentEncoder(
        G_ch=args.content_start_channel, resolution=args.content_image_size[0]
    )

    if optimize:
        # Channels last format (memory optimization only)
        content_image_encoder = content_image_encoder.to(
            memory_format=torch.channels_last
        )

    print("Get CG-GAN Content Encoder!")
    return content_image_encoder


@lru_cache(maxsize=1)
def build_content_encoder_cached(args, optimize=True):
    """
    Cached version of content encoder
    """
    return build_content_encoder_optimized(args, optimize)


def build_scr_optimized(args, optimize=True):
    """
    Build SCR module with memory optimizations only
    """
    scr = SCR(
        temperature=args.temperature, mode=args.mode, image_size=args.scr_image_size
    )

    if optimize:
        # Channels last format (memory optimization only)
        scr = scr.to(memory_format=torch.channels_last)

    print("Loaded SCR module for supervision successfully!")
    return scr


@lru_cache(maxsize=1)
def build_scr_cached(args, optimize=True):
    """
    Cached version of SCR
    """
    return build_scr_optimized(args, optimize)


def build_ddpm_scheduler_optimized(args):
    """
    Build DDPMScheduler with inference optimizations
    These don't affect model weights, only inference behavior
    """
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule=args.beta_scheduler,
        trained_betas=None,
        variance_type="fixed_small",
        clip_sample=True,
        prediction_type="epsilon",  # Explicitly set for clarity
        # Inference optimizations (don't affect weights)
        timestep_spacing="linspace",  # Keep original for weight compatibility
        steps_offset=0,
        thresholding=False,  # Disable for speed (inference only)
        dynamic_thresholding_ratio=0.995,
        sample_max_value=1.0,
    )
    return ddpm_scheduler


def build_ddpm_scheduler_fast(args):
    """
    Fast scheduler for quick inference
    WARNING: May affect output quality but doesn't touch model weights
    """
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,  # Keep same total steps for compatibility
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule=args.beta_scheduler,
        trained_betas=None,
        variance_type="fixed_small",
        clip_sample=True,
        prediction_type="epsilon",
        # These change inference behavior only
        timestep_spacing="leading"
        if getattr(args, "fast_inference", False)
        else "linspace",
        steps_offset=0,
        thresholding=False,  # Disable for speed
        dynamic_thresholding_ratio=0.995,
        sample_max_value=1.0,
    )
    return ddpm_scheduler


# Backward compatibility - these match original build.py exactly
def build_unet(args):
    """Original build_unet for backward compatibility"""
    return build_unet_optimized(args, optimize=False)


def build_style_encoder(args):
    """Original build_style_encoder for backward compatibility"""
    return build_style_encoder_optimized(args, optimize=False)


def build_content_encoder(args):
    """Original build_content_encoder for backward compatibility"""
    return build_content_encoder_optimized(args, optimize=False)


def build_scr(args):
    """Original build_scr for backward compatibility"""
    return build_scr_optimized(args, optimize=False)


def build_ddpm_scheduler(args):
    """Original build_ddpm_scheduler for backward compatibility"""
    return build_ddpm_scheduler_optimized(args)
