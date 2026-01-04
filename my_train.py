"""
FontDiffuser training script with multi-phase support.

Supports both Phase 1 (basic diffusion) and Phase 2 (with style-content recognition).
"""

import logging
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from torchvision import transforms
from tqdm.auto import tqdm

from configs.fontdiffuser import get_parser
from dataset.collate_fn import CollateFN
from dataset.font_dataset import FontDataset
from src import (
    ContentPerceptualLoss,
    FontDiffuserModel,
    StyleTransformationModule,

    build_content_encoder,
    build_ddpm_scheduler,
    build_scr,
    build_style_encoder,
    build_unet,
)
from utilities import (
    find_checkpoint,
    load_model_checkpoint,
    save_model_checkpoint,
    get_hf_bar,
)
from utils import (
    normalize_mean_std,
    reNormalize_img,
    save_args_to_yaml,
    x0_from_epsilon,
)

logger = get_logger(__name__)


def setup_logging(output_dir: Path) -> None:
    """Configure logging to file and console.

    Args:
        output_dir: Directory for log file
    """
    log_file = output_dir / "training.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def parse_args():
    """Parse and validate command line arguments.

    Returns:
        Parsed arguments with validated image sizes
    """
    parser = get_parser()
    args = parser.parse_args()

    # Handle distributed training rank
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1:
        args.local_rank = env_local_rank

    # Convert image sizes to tuples
    args.style_image_size = (args.style_image_size, args.style_image_size)
    args.content_image_size = (args.content_image_size, args.content_image_size)

    return args


def load_phase1_checkpoints(model_components: dict, ckpt_dir: str) -> None:
    """Load Phase 1 model checkpoints.

    Args:
        model_components: Dictionary of model components to load
        ckpt_dir: Directory containing checkpoints

    Raises:
        FileNotFoundError: If required checkpoint is missing
    """
    logger.info("Loading Phase 1 checkpoints...")

    for name, component in model_components.items():
        ckpt_path = find_checkpoint(ckpt_dir, name)
        component.load_state_dict(load_model_checkpoint(ckpt_path))
        logger.info(f"Loaded {name} from {ckpt_path}")


def create_transforms(args):
    """Create image transformation pipelines.

    Args:
        args: Training arguments

    Returns:
        Tuple of (content_transforms, style_transforms, target_transforms)
    """
    content_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.content_image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    style_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.style_image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    target_transforms = transforms.Compose(
        [
            transforms.Resize(
                (args.resolution, args.resolution),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    return content_transforms, style_transforms, target_transforms


def apply_classifier_free_guidance(
    content_images: torch.Tensor, style_images: torch.Tensor, drop_prob: float
) -> None:
    """Apply classifier-free guidance by masking inputs.

    Args:
        content_images: Content image batch (modified in-place)
        style_images: Style image batch (modified in-place)
        drop_prob: Probability of dropping conditioning
    """
    bsz = content_images.shape[0]
    context_mask = torch.bernoulli(torch.zeros(bsz) + drop_prob)

    for i, should_drop in enumerate(context_mask):
        if should_drop:
            content_images[i] = 1.0
            style_images[i] = 1.0


def compute_losses(
    noise_pred: torch.Tensor,
    noise: torch.Tensor,
    offset_out_sum: torch.Tensor,
    noisy_target_images: torch.Tensor,
    nonorm_target_images: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler,
    perceptual_loss: ContentPerceptualLoss,
    args,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Compute all training losses.

    Args:
        noise_pred: Predicted noise from model
        noise: Ground truth noise
        offset_out_sum: Offset prediction sum
        noisy_target_images: Noisy target images
        nonorm_target_images: Non-normalized target images
        timesteps: Diffusion timesteps
        noise_scheduler: DDPM scheduler
        perceptual_loss: Perceptual loss module
        args: Training arguments
        device: Compute device

    Returns:
        Tuple of (total_loss, loss_dict)
    """
    # Diffusion loss
    diff_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

    # Offset loss
    offset_loss = offset_out_sum / 2

    # Perceptual loss
    pred_original_sample_norm = x0_from_epsilon(
        scheduler=noise_scheduler,
        noise_pred=noise_pred,
        x_t=noisy_target_images,
        timesteps=timesteps,
    )
    pred_original_sample = reNormalize_img(pred_original_sample_norm)
    norm_pred_ori = normalize_mean_std(pred_original_sample)
    norm_target_ori = normalize_mean_std(nonorm_target_images)

    percep_loss = perceptual_loss.calculate_loss(
        generated_images=norm_pred_ori,
        target_images=norm_target_ori,
        device=device,
    )

    # Combine losses
    total_loss = (
        diff_loss
        + args.perceptual_coefficient * percep_loss
        + args.offset_coefficient * offset_loss
    )

    loss_dict = {
        "diff_loss": diff_loss.item(),
        "percep_loss": percep_loss.item(),
        "offset_loss": offset_loss.item(),
    }

    return total_loss, loss_dict, pred_original_sample_norm


def compute_phase2_loss(
    pred_original_sample_norm: torch.Tensor,
    target_images: torch.Tensor,
    neg_images: torch.Tensor,
    scr,
    args,
) -> torch.Tensor:
    """Compute Phase 2 style-content loss.

    Args:
        pred_original_sample_norm: Predicted original sample
        target_images: Target images
        neg_images: Negative style images
        scr: Style-content recognition module
        args: Training arguments

    Returns:
        Style-content loss
    """
    sample_emb, pos_emb, neg_emb = scr(
        pred_original_sample_norm,
        target_images,
        neg_images,
        nce_layers=args.nce_layers,
    )

    sc_loss = scr.calculate_nce_loss(
        sample_s=sample_emb,
        pos_s=pos_emb,
        neg_s=neg_emb,
    )

    return sc_loss


def save_checkpoint(
    model: FontDiffuserModel,
    scr: Optional[torch.nn.Module],
    accelerator: Accelerator,
    output_dir: str,
    global_step: int,
    is_phase2: bool,
) -> None:
    """Save model checkpoint.

    Args:
        model: FontDiffuser model
        scr: Style-content recognition module (optional)
        accelerator: Accelerator instance
        output_dir: Output directory
        global_step: Current global step
        is_phase2: Whether in Phase 2 training
    """
    save_dir = Path(output_dir) / f"global_step_{global_step}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Unwrap model from DDP/FSDP wrapper
    unwrapped_model = accelerator.unwrap_model(model)

    # Save individual components with safetensors format
    save_model_checkpoint(
        unwrapped_model.config.unet.state_dict(), save_dir / "unet.safetensors"
    )
    save_model_checkpoint(
        unwrapped_model.config.style_encoder.state_dict(),
        save_dir / "style_encoder.safetensors",
    )
    save_model_checkpoint(
        unwrapped_model.config.content_encoder.state_dict(),
        save_dir / "content_encoder.safetensors",
    )
    save_model_checkpoint(
        unwrapped_model.state_dict(), save_dir / "total_model.safetensors"
    )

    # Save SCR if Phase 2
    if is_phase2 and scr is not None:
        save_model_checkpoint(
            scr.state_dict(),
            save_dir / "scr.safetensors",
        )

    logger.info(f"Saved checkpoint at step {global_step} to {save_dir}")


def train_step(
    model: FontDiffuserModel,
    samples: dict,
    noise_scheduler,
    perceptual_loss: ContentPerceptualLoss,
    scr: Optional[torch.nn.Module],
    args,
    accelerator: Accelerator,
) -> tuple[torch.Tensor, dict]:
    """Execute single training step."""
    model.train()

    content_images = samples["content_image"]
    style_images = samples["style_image"]
    target_images = samples["target_image"]
    nonorm_target_images = samples["nonorm_target_image"]

    # Sample noise and timesteps
    noise = torch.randn_like(target_images)
    bsz = target_images.shape[0]
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (bsz,),
        device=target_images.device,
    ).long()

    # Add noise (forward diffusion)
    noisy_target_images = noise_scheduler.add_noise(target_images, noise, timesteps)

    # Apply classifier-free guidance
    apply_classifier_free_guidance(content_images, style_images, args.drop_prob)

    # ✅ PREPARE SOURCE STYLE IMAGES FOR STYLE TRANSFORMATION
    source_style_images = None
    if getattr(args, 'enable_style_transform', False) and 'source_style_image' in samples:
        source_style_images = samples["source_style_image"]
        apply_classifier_free_guidance(
            torch.zeros_like(source_style_images),
            source_style_images,
            args.drop_prob
        )

    # Forward pass
    noise_pred, offset_out_sum, style_transform_feature = model(
        x_t=noisy_target_images,
        timesteps=timesteps,
        style_images=style_images,
        content_images=content_images,
        content_encoder_downsample_size=args.content_encoder_downsample_size,
        source_style_images=source_style_images,  # ✅ ADD THIS
    )

    # Compute losses
    loss, loss_dict, pred_original_sample_norm = compute_losses(
        noise_pred=noise_pred,
        noise=noise,
        offset_out_sum=offset_out_sum,
        noisy_target_images=noisy_target_images,
        nonorm_target_images=nonorm_target_images,
        timesteps=timesteps,
        noise_scheduler=noise_scheduler,
        perceptual_loss=perceptual_loss,
        args=args,
        device=target_images.device,
    )

    # ✅ ADD STYLE TRANSFORMATION LOSS IF APPLICABLE
    if getattr(args, 'enable_style_transform', False) and style_transform_feature is not None:
        style_transform_loss = F.mse_loss(
            style_transform_feature,
            torch.zeros_like(style_transform_feature),
            reduction='mean'
        )
        loss += getattr(args, 'style_transform_coefficient', 0.1) * style_transform_loss
        loss_dict['style_transform_loss'] = style_transform_loss.item()

    # Add Phase 2 loss if applicable
    if args.phase_2 and scr is not None:
        sc_loss = compute_phase2_loss(
            pred_original_sample_norm=pred_original_sample_norm,
            target_images=target_images,
            neg_images=samples["neg_images"],
            scr=scr,
            args=args,
        )
        loss += args.sc_coefficient * sc_loss
        loss_dict["sc_loss"] = sc_loss.item()

    return loss, loss_dict

def train(
    model: FontDiffuserModel,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    noise_scheduler,
    perceptual_loss: ContentPerceptualLoss,
    scr: Optional[torch.nn.Module],
    accelerator: Accelerator,
    args,
) -> None:
    """Main training loop.

    Args:
        model: FontDiffuser model
        train_dataloader: Training data loader
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        noise_scheduler: DDPM scheduler
        perceptual_loss: Perceptual loss module
        scr: Style-content recognition module (optional)
        accelerator: Accelerator instance
        args: Training arguments
    """
    # Calculate training parameters
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Setup progress bar
    progress_bar = get_hf_bar(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )

    global_step = 0
    train_loss = 0.0

    for epoch in range(num_train_epochs):
        for step, samples in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # Training step
                loss, loss_dict = train_step(
                    model=model,
                    samples=samples,
                    noise_scheduler=noise_scheduler,
                    perceptual_loss=perceptual_loss,
                    scr=scr,
                    args=args,
                    accelerator=accelerator,
                )

                # Gather loss across processes
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backward pass
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Log metrics
                accelerator.log({"train_loss": train_loss}, step=global_step)

                if global_step % args.log_interval == 0:
                    logger.info(
                        f"Step {global_step}: loss={train_loss:.4f}, "
                        f"lr={lr_scheduler.get_last_lr()[0]:.6f}"
                    )

                train_loss = 0.0

                # Save checkpoint
                if (
                    accelerator.is_main_process
                    and global_step % args.ckpt_interval == 0
                ):
                    save_checkpoint(
                        model=model,
                        scr=scr,
                        accelerator=accelerator,
                        output_dir=args.output_dir,
                        global_step=global_step,
                        is_phase2=args.phase_2,
                    )

            # Update progress bar
            progress_bar.set_postfix(
                loss=loss.detach().item(), lr=lr_scheduler.get_last_lr()[0]
            )

            # Check if training is complete
            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    progress_bar.close()


def main():
    """Main training entry point."""
    args = parse_args()

    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=f"{args.output_dir}/{args.logging_dir}",
    )

    # Setup output directory and logging
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        setup_logging(Path(args.output_dir))

    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)

    # Build models
    unet = build_unet(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    noise_scheduler = build_ddpm_scheduler(args)

    # ✅ BUILD STYLE TRANSFORMATION MODULE
    style_transform_module = None
    if getattr(args, 'enable_style_transform', False):
        logger.info("Building Style Transformation Module...")
        style_transform_module = StyleTransformationModule(
            num_scales=getattr(args, 'num_scales', 4),
            feature_dim=getattr(args, 'feature_dim', 512),
            hidden_dim=getattr(args, 'hidden_dim', 256),
            num_heads=getattr(args, 'num_heads', 8),
            ffn_dim=getattr(args, 'ffn_dim', 2048),
        )
        logger.info("✓ Style Transformation Module built successfully")

    # Load Phase 1 checkpoints if provided
    if args.phase_1_ckpt_dir is not None:
        load_phase1_checkpoints(
            {
                "unet": unet,
                "style_encoder": style_encoder,
                "content_encoder": content_encoder,
            },
            args.phase_1_ckpt_dir,
        )

    # Build FontDiffuser model
    model = FontDiffuserModel(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder,
        style_transform_module=style_transform_module,  # ✅ ADD THIS
    )

    # Build perceptual loss
    perceptual_loss = ContentPerceptualLoss()

    # Build SCR for Phase 2
    scr = None
    if args.phase_2:
        scr = build_scr(args=args)
        if args.scr_ckpt_path:
            try:
                scr.load_state_dict(load_model_checkpoint(args.scr_ckpt_path))
                logger.info(f"Loaded SCR from {args.scr_ckpt_path}")
            except FileNotFoundError:
                logger.warning("SCR checkpoint not found, using untrained SCR")
        scr.requires_grad_(False)

    # Create datasets
    content_tfm, style_tfm, target_tfm = create_transforms(args)
    train_dataset = FontDataset(
        args=args,
        phase="train",
        transforms=[content_tfm, style_tfm, target_tfm],
        scr=args.phase_2,
        include_source_style=getattr(args, 'enable_style_transform', False),
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        collate_fn=CollateFN(),
    )

    # Build optimizer
    if args.scale_lr:
        args.learning_rate *= (
            args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Build learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare for distributed training
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Move SCR to device after preparation
    if args.phase_2 and scr is not None:
        scr = scr.to(accelerator.device)

    # Initialize trackers
    if accelerator.is_main_process:
        accelerator.init_trackers(args.experience_name)
        save_args_to_yaml(
            args=args,
            output_file=f"{args.output_dir}/{args.experience_name}_config.yaml",
        )

    # Train
    logger.info("Starting training...")
    logger.info(f"  Num examples: {len(train_dataset)}")
    logger.info(f"  Num batches per epoch: {len(train_dataloader)}")
    logger.info(f"  Total training steps: {args.max_train_steps}")
    logger.info(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"  Style Transform Module: {getattr(args, 'enable_style_transform', False)}")

    train(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        noise_scheduler=noise_scheduler,
        perceptual_loss=perceptual_loss,
        scr=scr,
        accelerator=accelerator,
        args=args,
    )

    accelerator.end_training()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()

"""Example:
python my_train.py \
    --enable_style_transform \
    --num_scales 4 \
    --feature_dim 512 \
    --style_transform_coefficient 0.1
"""