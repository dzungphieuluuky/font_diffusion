import os
import math
import time
import logging
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import (
    RandomAffine,
    ColorJitter,
    RandomRotation,
    GaussianBlur,
)

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler

from dataset.font_dataset import FontDataset
from dataset.collate_fn import CollateFN
from configs.fontdiffuser import get_parser
from src import (
    FontDiffuserModel,
    ContentPerceptualLoss,
    build_unet,
    build_style_encoder,
    build_content_encoder,
    build_ddpm_scheduler,
    build_scr,
)
from utils import (
    save_args_to_yaml,
    x0_from_epsilon,
    reNormalize_img,
    normalize_mean_std,
)

try:
    from safetensors.torch import load_file as load_safetensors

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("⚠️  safetensors not installed. Only .pth files will be supported.")


logger = get_logger(__name__)


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args


def validate(model, val_dataloader, noise_scheduler, accelerator, args, global_step):
    """Run validation loop and return average validation loss"""
    model.eval()
    val_loss = 0.0
    num_batches = 0

    with torch.inference_mode():
        for val_samples in val_dataloader:
            content_images = val_samples["content_image"]
            style_images = val_samples["style_image"]
            target_images = val_samples["target_image"]

            noise = torch.randn_like(target_images)
            bsz = target_images.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.num_train_timesteps,
                (bsz,),
                device=target_images.device,
            ).long()
            noisy_target_images = noise_scheduler.add_noise(
                target_images, noise, timesteps
            )

            # Classifier-free validation (no masking)
            noise_pred, _ = model(
                x_t=noisy_target_images,
                timesteps=timesteps,
                style_images=style_images,
                content_images=content_images,
                content_encoder_downsample_size=args.content_encoder_downsample_size,
            )

            batch_val_loss = F.mse_loss(
                noise_pred.float(), noise.float(), reduction="mean"
            )
            val_loss += batch_val_loss.item()
            num_batches += 1

    avg_val_loss = val_loss / max(num_batches, 1)
    return avg_val_loss


def load_checkpoint_file(checkpoint_path: str, device="cpu"):
    """
    Load checkpoint from either .pth or .safetensors format

    Args:
        checkpoint_path: Path to checkpoint file (.pth or .safetensors)
        device: Device to load to

    Returns:
        Loaded state dict

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        ValueError: If unsupported file format
    """
    checkpoint_path = str(checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Checkpoint not found: {checkpoint_path}")

    file_ext = os.path.splitext(checkpoint_path)[1].lower()

    try:
        if file_ext == ".safetensors":
            if not SAFETENSORS_AVAILABLE:
                raise ValueError(
                    f"❌ safetensors format not supported.\n"
                    f"   Install: pip install safetensors\n"
                    f"   Or use .pth format instead"
                )

            print(f"  📂 Loading safetensors: {os.path.basename(checkpoint_path)}")
            state_dict = load_safetensors(checkpoint_path)
            return state_dict

        elif file_ext == ".pth":
            print(f"  📂 Loading .pth: {os.path.basename(checkpoint_path)}")
            state_dict = torch.load(checkpoint_path, map_location=device)
            return state_dict

        else:
            raise ValueError(
                f"❌ Unsupported checkpoint format: {file_ext}\n"
                f"   Supported: .pth, .safetensors\n"
                f"   Got: {checkpoint_path}"
            )

    except Exception as e:
        raise RuntimeError(f"❌ Error loading checkpoint {checkpoint_path}: {e}")


def load_phase_1_checkpoint(args, unet, style_encoder, content_encoder):
    """
    Load Phase 1 checkpoint (unet, style_encoder, content_encoder)
    Handles both .pth and .safetensors formats

    Args:
        args: Training arguments
        unet: UNet model to load state into
        style_encoder: Style encoder model to load state into
        content_encoder: Content encoder model to load state into
    """
    if not args.phase_2:
        return

    ckpt_dir = Path(args.phase_1_ckpt_dir)

    if not ckpt_dir.exists():
        raise ValueError(f"❌ Phase 1 checkpoint directory not found: {ckpt_dir}")

    print(f"\n📥 Loading Phase 1 checkpoint from: {ckpt_dir}")
    print("=" * 60)

    # Define checkpoint files to load (try both formats)
    checkpoint_configs = [
        ("unet", unet),
        ("style_encoder", style_encoder),
        ("content_encoder", content_encoder),
    ]

    for model_name, model in checkpoint_configs:
        # Try .safetensors first, then .pth
        safetensors_path = ckpt_dir / f"{model_name}.safetensors"
        pth_path = ckpt_dir / f"{model_name}.pth"

        checkpoint_path = None

        if safetensors_path.exists():
            checkpoint_path = safetensors_path
        elif pth_path.exists():
            checkpoint_path = pth_path
        else:
            raise FileNotFoundError(
                f"❌ {model_name} checkpoint not found!\n"
                f"   Tried: {safetensors_path}\n"
                f"          {pth_path}"
            )

        try:
            state_dict = load_checkpoint_file(str(checkpoint_path))
            model.load_state_dict(state_dict)
            print(f"  ✓ {model_name}: {checkpoint_path.name}")
        except Exception as e:
            raise RuntimeError(f"❌ Error loading {model_name}: {e}")

    print("=" * 60)
    print("✓ Phase 1 checkpoint loaded successfully\n")


def load_scr_checkpoint(scr_ckpt_path: str, scr_model):
    """
    Load SCR (Style-Content contrastive) module checkpoint
    Handles both .pth and .safetensors formats

    Args:
        scr_ckpt_path: Path to SCR checkpoint
        scr_model: SCR model to load state into
    """
    scr_ckpt_path = Path(scr_ckpt_path)

    if not scr_ckpt_path.exists():
        raise FileNotFoundError(f"❌ SCR checkpoint not found: {scr_ckpt_path}")

    print(f"📥 Loading SCR module from: {scr_ckpt_path}")

    try:
        state_dict = load_checkpoint_file(str(scr_ckpt_path))
        scr_model.load_state_dict(state_dict)
        print(f"  ✓ SCR module loaded: {scr_ckpt_path.name}\n")
    except Exception as e:
        raise RuntimeError(f"❌ Error loading SCR module: {e}")


def save_checkpoint(model, accelerator, args, global_step, is_best=False):
    """Save model checkpoint in both .pth and .safetensors format"""
    if is_best:
        save_dir = f"{args.output_dir}/best_checkpoint"
    else:
        save_dir = f"{args.output_dir}/global_step_{global_step}"

    os.makedirs(save_dir, exist_ok=True)

    # Save as .pth (original format)
    torch.save(model.unet.state_dict(), f"{save_dir}/unet.pth")
    torch.save(model.style_encoder.state_dict(), f"{save_dir}/style_encoder.pth")
    torch.save(model.content_encoder.state_dict(), f"{save_dir}/content_encoder.pth")
    torch.save(model, f"{save_dir}/total_model.pth")

    # ✅ ALSO save as .safetensors if available
    if SAFETENSORS_AVAILABLE:
        try:
            from safetensors.torch import save_file as save_safetensors

            save_safetensors(model.unet.state_dict(), f"{save_dir}/unet.safetensors")
            save_safetensors(
                model.style_encoder.state_dict(),
                f"{save_dir}/style_encoder.safetensors",
            )
            save_safetensors(
                model.content_encoder.state_dict(),
                f"{save_dir}/content_encoder.safetensors",
            )
        except Exception as e:
            print(f"⚠️  Could not save safetensors format: {e}")

    logging.info(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] "
        f"Saved checkpoint at global step {global_step}"
    )
    print(f"✓ Saved checkpoint at global step {global_step}")


def main():
    args = get_args()

    # Add missing arguments with defaults
    if not hasattr(args, "val_interval"):
        args.val_interval = 100  # Validate every 100 steps
    # ===== Validation: Check checkpoint interval vs max_train_steps =====
    if args.ckpt_interval > args.max_train_steps:
        raise ValueError(
            f"❌ ERROR: ckpt_interval ({args.ckpt_interval}) is larger than "
            f"max_train_steps ({args.max_train_steps})!\n"
            f"   Set ckpt_interval to a value < {args.max_train_steps}\n"
            f"   Suggested: ckpt_interval = {args.max_train_steps // 4}"
        )

    if args.max_train_steps % args.ckpt_interval != 0:
        recommended = (args.max_train_steps // args.ckpt_interval) * args.ckpt_interval
        print(
            f"⚠️  WARNING: max_train_steps ({args.max_train_steps}) is not divisible by "
            f"ckpt_interval ({args.ckpt_interval}).\n"
            f"   Last checkpoint will be at step {recommended}, final model will be saved separately."
        )
        logging.warning(
            f"max_train_steps ({args.max_train_steps}) not divisible by "
            f"ckpt_interval ({args.ckpt_interval})"
        )

    logging_dir = f"{args.output_dir}/{args.logging_dir}"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        filename=f"{args.output_dir}/fontdiffuser_training.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Set training seed
    if args.seed is not None:
        set_seed(args.seed)

    # Load model and noise_scheduler
    unet = build_unet(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    noise_scheduler = build_ddpm_scheduler(args)

    # ✅ FIXED: Load Phase 1 checkpoint with proper error handling
    if args.phase_2:
        try:
            load_phase_1_checkpoint(args, unet, style_encoder, content_encoder)
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"\n{e}")
            logging.error(str(e))
            raise

    model = FontDiffuserModel(
        unet=unet, style_encoder=style_encoder, content_encoder=content_encoder
    )

    # Build content perceptual loss
    perceptual_loss = ContentPerceptualLoss()

    scr = None
    if args.phase_2:
        scr = build_scr(args=args)
        try:
            load_scr_checkpoint(args.scr_ckpt_path, scr)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"\n{e}")
            logging.error(str(e))
            raise

        scr.requires_grad_(False)
        logging.info(f"Loaded SCR module from: {args.scr_ckpt_path}")

    # Load the training dataset with augmentation
    content_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.content_image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            RandomRotation(degrees=5),
            RandomAffine(degrees=0, translate=(0.05, 0.05), shear=(-5, 5)),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
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
            RandomRotation(degrees=5),
            ColorJitter(brightness=0.15, contrast=0.15),
            GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
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
            RandomRotation(degrees=3),
            ColorJitter(brightness=0.1, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    train_font_dataset = FontDataset(
        args=args,
        phase="train",
        transforms=[content_transforms, style_transforms, target_transforms],
        scr=args.phase_2,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_font_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        collate_fn=CollateFN(),
    )

    # Load validation dataset (without augmentation)
    val_content_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.content_image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    val_style_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.style_image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    val_target_transforms = transforms.Compose(
        [
            transforms.Resize(
                (args.resolution, args.resolution),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    val_font_dataset = FontDataset(
        args=args,
        phase="val_unseen_both",
        transforms=[
            val_content_transforms,
            val_style_transforms,
            val_target_transforms,
        ],
        scr=args.phase_2,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_font_dataset,
        shuffle=False,
        batch_size=args.train_batch_size,
        collate_fn=CollateFN(),
    )

    # Build optimizer and learning rate scheduler
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
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

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Accelerate preparation
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    val_dataloader = accelerator.prepare(val_dataloader)

    # Move SCR module to target device
    if scr is not None:
        scr = scr.to(accelerator.device)

    # Initialize trackers and save config
    if accelerator.is_main_process:
        accelerator.init_trackers(
            args.experience_name,
            config={
                "max_train_steps": args.max_train_steps,
                "train_batch_size": args.train_batch_size,
                "learning_rate": args.learning_rate,
                "perceptual_coefficient": args.perceptual_coefficient,
                "offset_coefficient": args.offset_coefficient,
                "sc_coefficient": args.sc_coefficient if args.phase_2 else 0,
                "phase_2": args.phase_2,
                "seed": args.seed,
            },
        )
        save_args_to_yaml(
            args=args,
            output_file=f"{args.output_dir}/{args.experience_name}_config.yaml",
        )

    # Setup progress bar
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")

    # Training epoch setup
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    best_val_loss = float("inf")

    logging.info(
        f"Starting training with {num_train_epochs} epochs, "
        f"{num_update_steps_per_epoch} steps per epoch"
    )
    logging.info(
        f"Max training steps: {args.max_train_steps}, "
        f"Checkpoint interval: {args.ckpt_interval}"
    )

    for epoch in range(num_train_epochs):
        epoch_train_loss = 0.0
        epoch_diff_loss = 0.0
        epoch_percep_loss = 0.0
        epoch_offset_loss = 0.0
        epoch_sc_loss = 0.0
        epoch_steps = 0

        for step, samples in enumerate(train_dataloader):
            model.train()
            content_images = samples["content_image"]
            style_images = samples["style_image"]
            target_images = samples["target_image"]
            nonorm_target_images = samples["nonorm_target_image"]

            with accelerator.accumulate(model):
                # Sample noise
                noise = torch.randn_like(target_images)
                bsz = target_images.shape[0]

                # Sample random timesteps
                timesteps = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=target_images.device,
                ).long()

                # Add noise (forward diffusion)
                noisy_target_images = noise_scheduler.add_noise(
                    target_images, noise, timesteps
                )

                # Classifier-free guidance training strategy
                context_mask = torch.bernoulli(torch.zeros(bsz) + args.drop_prob)
                for i, mask_value in enumerate(context_mask):
                    if mask_value == 1:
                        content_images[i, :, :, :] = 1
                        style_images[i, :, :, :] = 1

                # Predict noise and offset
                noise_pred, offset_out_sum = model(
                    x_t=noisy_target_images,
                    timesteps=timesteps,
                    style_images=style_images,
                    content_images=content_images,
                    content_encoder_downsample_size=args.content_encoder_downsample_size,
                )

                # ============================================================================
                # ✅ CALCULATE INDIVIDUAL LOSSES (for logging)
                # ============================================================================

                # Diffusion loss
                diff_loss = F.mse_loss(
                    noise_pred.float(), noise.float(), reduction="mean"
                )
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
                    device=target_images.device,
                )

                # Dynamic loss coefficient scheduling
                progress_ratio = global_step / args.max_train_steps
                if progress_ratio < 0.5:
                    percep_coeff = args.perceptual_coefficient * (progress_ratio / 0.5)
                else:
                    percep_coeff = args.perceptual_coefficient

                # Combine losses
                loss = (
                    diff_loss
                    + percep_coeff * percep_loss
                    + args.offset_coefficient * offset_loss
                )

                # ============================================================================
                # Phase 2: Style-content contrastive loss
                # ============================================================================
                sc_loss_value = 0.0  # Initialize for logging

                if args.phase_2 and scr is not None:
                    neg_images = samples["neg_images"]
                    (
                        sample_style_embeddings,
                        pos_style_embeddings,
                        neg_style_embeddings,
                    ) = scr(
                        pred_original_sample_norm,
                        target_images,
                        neg_images,
                        nce_layers=args.nce_layers,
                    )
                    sc_loss_value = scr.calculate_nce_loss(
                        sample_s=sample_style_embeddings,
                        pos_s=pos_style_embeddings,
                        neg_s=neg_style_embeddings,
                    )
                    loss += args.sc_coefficient * sc_loss_value

                # Gather losses for logging
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                avg_diff_loss = accelerator.gather(
                    diff_loss.repeat(args.train_batch_size)
                ).mean()
                avg_percep_loss = accelerator.gather(
                    percep_loss.repeat(args.train_batch_size)
                ).mean()
                avg_offset_loss = accelerator.gather(
                    offset_loss.repeat(args.train_batch_size)
                ).mean()

                epoch_train_loss += avg_loss.item() / args.gradient_accumulation_steps
                epoch_diff_loss += (
                    avg_diff_loss.item() / args.gradient_accumulation_steps
                )
                epoch_percep_loss += (
                    avg_percep_loss.item() / args.gradient_accumulation_steps
                )
                epoch_offset_loss += (
                    avg_offset_loss.item() / args.gradient_accumulation_steps
                )

                if args.phase_2 and scr is not None:
                    avg_sc_loss = accelerator.gather(
                        sc_loss_value.repeat(args.train_batch_size)
                    ).mean()
                    epoch_sc_loss += (
                        avg_sc_loss.item() / args.gradient_accumulation_steps
                    )

                # Backpropagation
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # ✅ Log gradient norm
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # ============================================================================
            # ✅ LOGGING TO W&B
            # ============================================================================

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                epoch_steps += 1

                # Log to W&B at every step
                if accelerator.is_main_process:
                    wandb_logs = {
                        "train/total_loss": epoch_train_loss / epoch_steps,
                        "train/diff_loss": epoch_diff_loss / epoch_steps,
                        "train/perceptual_loss": epoch_percep_loss / epoch_steps,
                        "train/offset_loss": epoch_offset_loss / epoch_steps,
                        "train/learning_rate": lr_scheduler.get_last_lr()[0],
                        "train/gradient_norm": grad_norm.item()
                        if isinstance(grad_norm, torch.Tensor)
                        else grad_norm,
                        "epoch": epoch,
                    }

                    # Add Phase 2 loss if applicable
                    if args.phase_2 and scr is not None:
                        wandb_logs["train/sc_loss"] = epoch_sc_loss / epoch_steps

                    accelerator.log(wandb_logs, step=global_step)

                if accelerator.is_main_process:
                    # Save checkpoint at intervals
                    if global_step % args.ckpt_interval == 0:
                        save_checkpoint(
                            model, accelerator, args, global_step, is_best=False
                        )

                    # Validation and best model selection
                    if global_step % args.val_interval == 0:
                        val_loss = validate(
                            model,
                            val_dataloader,
                            noise_scheduler,
                            accelerator,
                            args,
                            global_step,
                        )

                        # ✅ Log validation loss and metrics
                        val_logs = {
                            "val/loss": val_loss,
                        }
                        accelerator.log(val_logs, step=global_step)

                        logging.info(
                            f"Global Step {global_step} => val_loss = {val_loss:.6f}"
                        )

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            save_checkpoint(
                                model, accelerator, args, global_step, is_best=True
                            )
                            logging.info(
                                f"New best model saved with val_loss = {best_val_loss:.6f}"
                            )

                            # Log best model update
                            accelerator.log(
                                {"val/best_loss": best_val_loss}, step=global_step
                            )

            # Progress bar update
            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "epoch": epoch,
            }

            if global_step % args.log_interval == 0:
                logging.info(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] "
                    f"Global Step {global_step} => "
                    f"total_loss = {epoch_train_loss / epoch_steps:.6f}, "
                    f"diff_loss = {epoch_diff_loss / epoch_steps:.6f}, "
                    f"percep_loss = {epoch_percep_loss / epoch_steps:.6f}, "
                    f"offset_loss = {epoch_offset_loss / epoch_steps:.6f}"
                )

            progress_bar.set_postfix(**logs)

            # Exit if max steps reached
            if global_step >= args.max_train_steps:
                break

    # ===== Save final checkpoint =====
    if accelerator.is_main_process:
        save_checkpoint(model, accelerator, args, global_step, is_best=False)
        logging.info(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] "
            f"Training completed. Saved final checkpoint at global step {global_step}"
        )
        print(
            f"\n✓ Training completed! Final checkpoint saved at global step {global_step}"
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
