import os
import math
import time
import logging
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import RandomAffine, ColorJitter, RandomRotation, GaussianBlur

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler

from dataset.font_dataset import FontDataset
from dataset.collate_fn import CollateFN
from configs.fontdiffuser import get_parser
from src import (FontDiffuserModel,
                 ContentPerceptualLoss,
                 build_unet,
                 build_style_encoder,
                 build_content_encoder,
                 build_ddpm_scheduler,
                 build_scr)
from utils import (save_args_to_yaml,
                   x0_from_epsilon, 
                   reNormalize_img, 
                   normalize_mean_std)


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
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), 
                                     device=target_images.device).long()
            noisy_target_images = noise_scheduler.add_noise(target_images, noise, timesteps)
            
            # Classifier-free validation (no masking)
            noise_pred, _ = model(
                x_t=noisy_target_images, 
                timesteps=timesteps, 
                style_images=style_images,
                content_images=content_images,
                content_encoder_downsample_size=args.content_encoder_downsample_size)
            
            batch_val_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            val_loss += batch_val_loss.item()
            num_batches += 1
    
    avg_val_loss = val_loss / max(num_batches, 1)
    return avg_val_loss


def save_checkpoint(model, accelerator, args, global_step, is_best=False):
    """Save model checkpoint"""
    if is_best:
        save_dir = f"{args.output_dir}/best_checkpoint"
    else:
        save_dir = f"{args.output_dir}/global_step_{global_step}"
    
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.unet.state_dict(), f"{save_dir}/unet.pth")
    torch.save(model.style_encoder.state_dict(), f"{save_dir}/style_encoder.pth")
    torch.save(model.content_encoder.state_dict(), f"{save_dir}/content_encoder.pth")
    torch.save(model, f"{save_dir}/total_model.pth")
    
    logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] "
                f"Saved checkpoint at global step {global_step}")
    print(f"Saved checkpoint at global step {global_step}")


def main():
    args = get_args()

    # ===== Validation: Check checkpoint interval vs max_train_steps =====
    if args.ckpt_interval > args.max_train_steps:
        raise ValueError(
            f"❌ ERROR: ckpt_interval ({args.ckpt_interval}) is larger than "
            f"max_train_steps ({args.max_train_steps})!\n"
            f"   Set ckpt_interval to a value < {args.max_train_steps}\n"
            f"   Suggested: ckpt_interval = {args.max_train_steps // 4}")
    
    if args.max_train_steps % args.ckpt_interval != 0:
        recommended = (args.max_train_steps // args.ckpt_interval) * args.ckpt_interval
        print(f"⚠️  WARNING: max_train_steps ({args.max_train_steps}) is not divisible by "
              f"ckpt_interval ({args.ckpt_interval}).\n"
              f"   Last checkpoint will be at step {recommended}, final model will be saved separately.")
        logging.warning(
            f"max_train_steps ({args.max_train_steps}) not divisible by "
            f"ckpt_interval ({args.ckpt_interval})")

    logging_dir = f"{args.output_dir}/{args.logging_dir}"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        filename=f"{args.output_dir}/fontdiffuser_training.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')

    # Set training seed
    if args.seed is not None:
        set_seed(args.seed)

    # Load model and noise_scheduler
    unet = build_unet(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    noise_scheduler = build_ddpm_scheduler(args)
    
    # Load Phase 1 checkpoint for Phase 2 training
    if args.phase_2:
        logging.info(f"Loading Phase 1 checkpoint from: {args.phase_1_ckpt_dir}")
        unet.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/unet.pth"))
        style_encoder.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/style_encoder.pth"))
        content_encoder.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/content_encoder.pth"))
        print("✓ Phase 1 checkpoint loaded successfully")

    model = FontDiffuserModel(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder)

    # Build content perceptual loss
    perceptual_loss = ContentPerceptualLoss()

    # Load SCR module for Phase 2 supervision
    scr = None
    if args.phase_2:
        scr = build_scr(args=args)
        scr.load_state_dict(torch.load(args.scr_ckpt_path))
        scr.requires_grad_(False)
        logging.info(f"Loaded SCR module from: {args.scr_ckpt_path}")

    # Load the training dataset with augmentation
    content_transforms = transforms.Compose(
        [transforms.Resize(args.content_image_size, 
                           interpolation=transforms.InterpolationMode.BILINEAR),
         RandomRotation(degrees=5),
         RandomAffine(degrees=0, translate=(0.05, 0.05), shear=(-5, 5)),
         ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
         GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    
    style_transforms = transforms.Compose(
        [transforms.Resize(args.style_image_size, 
                           interpolation=transforms.InterpolationMode.BILINEAR),
         RandomRotation(degrees=5),
         ColorJitter(brightness=0.15, contrast=0.15),
         GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    
    target_transforms = transforms.Compose(
        [transforms.Resize((args.resolution, args.resolution), 
                           interpolation=transforms.InterpolationMode.BILINEAR),
         RandomRotation(degrees=3),
         ColorJitter(brightness=0.1, contrast=0.15),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    
    train_font_dataset = FontDataset(
        args=args,
        phase='train', 
        transforms=[
            content_transforms, 
            style_transforms, 
            target_transforms],
        scr=args.phase_2)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_font_dataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=CollateFN())
    
    # Load validation dataset (without augmentation)
    val_content_transforms = transforms.Compose(
        [transforms.Resize(args.content_image_size, 
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    
    val_style_transforms = transforms.Compose(
        [transforms.Resize(args.style_image_size, 
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    
    val_target_transforms = transforms.Compose(
        [transforms.Resize((args.resolution, args.resolution), 
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    
    val_font_dataset = FontDataset(
        args=args,
        phase='val_unseen_both',
        transforms=[
            val_content_transforms, 
            val_style_transforms, 
            val_target_transforms],
        scr=args.phase_2)
    
    val_dataloader = torch.utils.data.DataLoader(
        val_font_dataset, shuffle=False, batch_size=args.train_batch_size, collate_fn=CollateFN())
    
    # Build optimizer and learning rate scheduler
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps)

    # Accelerate preparation
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)
    
    val_dataloader = accelerator.prepare(val_dataloader)
    
    # Move SCR module to target device
    if scr is not None:
        scr = scr.to(accelerator.device)

    # Initialize trackers and save config
    if accelerator.is_main_process:
        accelerator.init_trackers(args.experience_name)
        save_args_to_yaml(args=args, output_file=f"{args.output_dir}/{args.experience_name}_config.yaml")

    # Setup progress bar
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Training epoch setup
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    best_val_loss = float('inf')
    
    logging.info(f"Starting training with {num_train_epochs} epochs, "
                f"{num_update_steps_per_epoch} steps per epoch")
    logging.info(f"Max training steps: {args.max_train_steps}, "
                f"Checkpoint interval: {args.ckpt_interval}")

    for epoch in range(num_train_epochs):
        train_loss = 0.0
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
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), 
                                         device=target_images.device).long()

                # Add noise (forward diffusion)
                noisy_target_images = noise_scheduler.add_noise(target_images, noise, timesteps)

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
                    content_encoder_downsample_size=args.content_encoder_downsample_size)
                
                # Calculate diffusion loss
                diff_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                offset_loss = offset_out_sum / 2
                
                # Calculate perceptual loss
                pred_original_sample_norm = x0_from_epsilon(
                    scheduler=noise_scheduler,
                    noise_pred=noise_pred,
                    x_t=noisy_target_images,
                    timesteps=timesteps)
                pred_original_sample = reNormalize_img(pred_original_sample_norm)
                norm_pred_ori = normalize_mean_std(pred_original_sample)
                norm_target_ori = normalize_mean_std(nonorm_target_images)
                percep_loss = perceptual_loss.calculate_loss(
                    generated_images=norm_pred_ori,
                    target_images=norm_target_ori,
                    device=target_images.device)
                
                # Dynamic loss coefficient scheduling (focus on diff_loss early, perceptual late)
                progress_ratio = global_step / args.max_train_steps
                if progress_ratio < 0.5:
                    percep_coeff = args.perceptual_coefficient * (progress_ratio / 0.5)
                else:
                    percep_coeff = args.perceptual_coefficient
                
                # Combine losses
                loss = diff_loss + \
                        percep_coeff * percep_loss + \
                        args.offset_coefficient * offset_loss
                
                # Phase 2: Add style-content contrastive loss
                if args.phase_2 and scr is not None:
                    neg_images = samples["neg_images"]
                    sample_style_embeddings, pos_style_embeddings, neg_style_embeddings = scr(
                        pred_original_sample_norm, 
                        target_images, 
                        neg_images, 
                        nce_layers=args.nce_layers)
                    sc_loss = scr.calculate_nce_loss(
                        sample_s=sample_style_embeddings,
                        pos_s=pos_style_embeddings,
                        neg_s=neg_style_embeddings)
                    loss += args.sc_coefficient * sc_loss

                # Gather losses for logging
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagation
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress after optimization step
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    # Save checkpoint at intervals
                    if global_step % args.ckpt_interval == 0:
                        save_checkpoint(model, accelerator, args, global_step, is_best=False)
                    
                    # Validation and best model selection
                    if global_step % args.val_interval == 0:
                        val_loss = validate(model, val_dataloader, noise_scheduler, 
                                          accelerator, args, global_step)
                        accelerator.log({"val_loss": val_loss}, step=global_step)
                        logging.info(f"Global Step {global_step} => val_loss = {val_loss:.6f}")
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            save_checkpoint(model, accelerator, args, global_step, is_best=True)
                            logging.info(f"New best model saved with val_loss = {best_val_loss:.6f}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if global_step % args.log_interval == 0:
                logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] "
                           f"Global Step {global_step} => train_loss = {loss:.6f}")
            progress_bar.set_postfix(**logs)
            
            # Exit if max steps reached
            if global_step >= args.max_train_steps:
                break

    # ===== Save final checkpoint =====
    if accelerator.is_main_process:
        save_checkpoint(model, accelerator, args, global_step, is_best=False)
        logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] "
                    f"Training completed. Saved final checkpoint at global step {global_step}")
        print(f"\n✓ Training completed! Final checkpoint saved at global step {global_step}")

    accelerator.end_training()


if __name__ == "__main__":
    main()