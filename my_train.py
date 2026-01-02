import os
import math
import time
import logging
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

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

def load_model_checkpoint(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file as safe_load
        state_dict = safe_load(checkpoint_path, device="cpu")
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    return state_dict


def save_model_checkpoint(model_state_dict, checkpoint_path: str):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import save_file as safe_save
        safe_save(model_state_dict, checkpoint_path)
    else:
        torch.save(model_state_dict, checkpoint_path)


def find_checkpoint(checkpoint_dir: str, checkpoint_name: str) -> str:
    """
    Find checkpoint file, preferring .safetensors over .pth
    Args:
        checkpoint_dir: Directory containing checkpoint
        checkpoint_name: Checkpoint name without extension (e.g., "unet")
    Returns:
        Full path to checkpoint file
    Raises:
        FileNotFoundError: If neither format exists
    """
    safetensors_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.safetensors")
    pth_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pth")
    
    if os.path.exists(safetensors_path):
        return safetensors_path
    elif os.path.exists(pth_path):
        return pth_path
    else:
        raise FileNotFoundError(
            f"Checkpoint not found for '{checkpoint_name}' in {checkpoint_dir}\n"
            f"  Expected: {safetensors_path} or {pth_path}"
        )

def main():
    args = get_args()

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
    )

    # Set training seed
    if args.seed is not None:
        set_seed(args.seed)

    # Load model and noise_scheduler
    unet = build_unet(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    noise_scheduler = build_ddpm_scheduler(args)
    
    # ========== LOAD PHASE 1 CHECKPOINTS ==========
    if args.phase_2:
        print("\n📦 Loading Phase 1 checkpoints...")
        try:
            unet_ckpt = find_checkpoint(args.phase_1_ckpt_dir, "unet")
            style_encoder_ckpt = find_checkpoint(args.phase_1_ckpt_dir, "style_encoder")
            content_encoder_ckpt = find_checkpoint(args.phase_1_ckpt_dir, "content_encoder")
            
            unet.load_state_dict(load_model_checkpoint(unet_ckpt))
            print(f"  ✓ Loaded unet from {unet_ckpt}")
            
            style_encoder.load_state_dict(load_model_checkpoint(style_encoder_ckpt))
            print(f"  ✓ Loaded style_encoder from {style_encoder_ckpt}")
            
            content_encoder.load_state_dict(load_model_checkpoint(content_encoder_ckpt))
            print(f"  ✓ Loaded content_encoder from {content_encoder_ckpt}")
            
        except FileNotFoundError as e:
            print(f"\n❌ Error loading Phase 1 checkpoint:")
            print(f"   {e}")
            raise

    model = FontDiffuserModel(
        unet=unet, style_encoder=style_encoder, content_encoder=content_encoder
    )

    # Build content perceptual Loss
    perceptual_loss = ContentPerceptualLoss()

    # Load SCR module for supervision
    scr = None
    if args.phase_2:
        scr = build_scr(args=args)
        try:
            scr.load_state_dict(load_model_checkpoint(args.scr_ckpt_path))
            print(f"  ✓ Loaded scr from {args.scr_ckpt_path}")
        except FileNotFoundError as e:
            print(f"\n⚠️  Warning: SCR checkpoint not found, using untrained SCR")
            print(f"   {e}")
        scr.requires_grad_(False)

    # Load the datasets
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

    # Build optimizer and learning rate
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

    # ✅ ACCELERATE PREPARATION - CORRECT ORDER
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # ✅ Move scr module to device AFTER accelerate.prepare()
    if args.phase_2 and scr is not None:
        scr = scr.to(accelerator.device)

    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.experience_name)
        save_args_to_yaml(
            args=args,
            output_file=f"{args.output_dir}/{args.experience_name}_config.yaml",
        )

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")

    # Convert to the training epoch
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    for epoch in range(num_train_epochs):
        train_loss = 0.0
        for step, samples in enumerate(train_dataloader):
            model.train()
            content_images = samples["content_image"]
            style_images = samples["style_image"]
            target_images = samples["target_image"]
            nonorm_target_images = samples["nonorm_target_image"]

            with accelerator.accumulate(model):
                # Sample noise that we'll add to the samples
                noise = torch.randn_like(target_images)
                bsz = target_images.shape[0]
                
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=target_images.device,
                )
                timesteps = timesteps.long()

                # Add noise to the target_images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_target_images = noise_scheduler.add_noise(
                    target_images, noise, timesteps
                )

                # Classifier-free training strategy
                context_mask = torch.bernoulli(torch.zeros(bsz) + args.drop_prob)
                for i, mask_value in enumerate(context_mask):
                    if mask_value == 1:
                        content_images[i, :, :, :] = 1
                        style_images[i, :, :, :] = 1

                # Predict the noise residual and compute loss
                noise_pred, offset_out_sum = model(
                    x_t=noisy_target_images,
                    timesteps=timesteps,
                    style_images=style_images,
                    content_images=content_images,
                    content_encoder_downsample_size=args.content_encoder_downsample_size,
                )
                
                diff_loss = F.mse_loss(
                    noise_pred.float(), noise.float(), reduction="mean"
                )
                offset_loss = offset_out_sum / 2

                # Output processing for content perceptual loss
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

                loss = (
                    diff_loss
                    + args.perceptual_coefficient * percep_loss
                    + args.offset_coefficient * offset_loss
                )

                # ✅ Phase 2 SC loss (with proper model unwrapping if needed)
                if args.phase_2 and scr is not None:
                    neg_images = samples["neg_images"]
                    # sc loss
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
                    sc_loss = scr.calculate_nce_loss(
                        sample_s=sample_style_embeddings,
                        pos_s=pos_style_embeddings,
                        neg_s=neg_style_embeddings,
                    )
                    loss += args.sc_coefficient * sc_loss

                # Gather the losses across all processes for logging
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # ✅ CHECKPOINT SAVING - ALWAYS UNWRAP MODEL
                if accelerator.is_main_process:
                    if global_step % args.ckpt_interval == 0:
                        save_dir = f"{args.output_dir}/global_step_{global_step}"
                        os.makedirs(save_dir, exist_ok=True)
                        
                        # ✅ UNWRAP MODEL BEFORE ACCESSING COMPONENTS
                        unwrapped_model = accelerator.unwrap_model(model)
                        
                        # Save individual components
                        # Prioritize save .safetensors format 
                        save_model_checkpoint(
                            unwrapped_model.unet.state_dict(),
                            f"{save_dir}/unet.safetensors"
                        )
                        save_model_checkpoint(
                            unwrapped_model.style_encoder.state_dict(),
                            f"{save_dir}/style_encoder.safetensors",
                        )
                        save_model_checkpoint(
                            unwrapped_model.content_encoder.state_dict(),
                            f"{save_dir}/content_encoder.safetensors",
                        )
                        
                        # Save full model
                        save_model_checkpoint(
                            unwrapped_model.state_dict(),
                            f"{save_dir}/total_model.safetensors"
                        )
                        
                        log_msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}] Save the checkpoint on global step {global_step}"
                        logging.info(log_msg)
                        print(f"✅ {log_msg}")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            if global_step % args.log_interval == 0:
                logging.info(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}] Global Step {global_step} => train_loss = {loss}"
                )
            progress_bar.set_postfix(**logs)

            # Quit
            if global_step >= args.max_train_steps:
                break

    accelerator.end_training()


if __name__ == "__main__":
    main()