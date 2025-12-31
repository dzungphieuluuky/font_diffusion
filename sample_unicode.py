import os
import cv2
import time
import random
import numpy as np
from PIL import Image
from typing import List, Optional

import torch
import torchvision.transforms as transforms
from accelerate.utils import set_seed

from src import (
    FontDiffuserDPMPipeline,
    FontDiffuserModelDPM,
    build_ddpm_scheduler,
    build_unet,
    build_content_encoder,
    build_style_encoder,
)
from utils import (
    ttf2im,
    load_ttf,
    is_char_in_font,
    save_args_to_yaml,
    save_single_image,
    save_image_with_content_style,
)


def arg_parse():
    from configs.fontdiffuser import get_parser

    parser = get_parser()
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument(
        "--controlnet",
        type=bool,
        default=False,
        help="If in demo mode, the controlnet can be added.",
    )
    parser.add_argument("--character_input", action="store_true")
    parser.add_argument("--content_character", type=str, default=None)
    parser.add_argument(
        "--content_characters",
        type=str,
        default=None,
        help="Comma-separated list of characters to generate",
    )
    parser.add_argument("--content_image_path", type=str, default=None)
    parser.add_argument("--style_image_path", type=str, default=None)
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument(
        "--save_image_dir", type=str, default=None, help="The saving directory."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ttf_path", type=str, default="ttf/KaiXinSongA.ttf")
    args = parser.parse_args()
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args


def create_transforms(args):
    """Create transforms once to reuse"""
    content_inference_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.content_image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    style_inference_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.style_image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    return content_inference_transforms, style_inference_transforms


def prepare_style_image(args, style_transforms):
    """Load and prepare style image once"""
    if not os.path.exists(args.style_image_path):
        raise FileNotFoundError(f"Style image not found: {args.style_image_path}")

    style_image = Image.open(args.style_image_path).convert("RGB")
    style_image_pil = style_image.copy()  # Keep original for saving

    # Transform to tensor
    style_image_tensor = style_transforms(style_image)[None, :]

    return style_image_pil, style_image_tensor


def prepare_content_image(args, content_transforms, character: str, ttf_font=None):
    """Prepare content image for a specific character"""
    if args.character_input:
        if not is_char_in_font(font_path=args.ttf_path, char=character):
            print(f"Character '{character}' not in font {args.ttf_path}")
            return None, None

        if ttf_font is None:
            ttf_font = load_ttf(ttf_path=args.ttf_path)

        content_image = ttf2im(font=ttf_font, char=character)
        content_image_pil = content_image.copy()
    else:
        # If not using character input, use provided image path
        content_image = Image.open(args.content_image_path).convert("RGB")
        content_image_pil = None

    # Transform to tensor
    content_image_tensor = content_transforms(content_image)[None, :]

    return content_image_pil, content_image_tensor


def load_fontdiffuser_pipeline(args):
    """Load the FontDiffuser pipeline once"""
    # Load the model state_dict
    unet = build_unet(args=args)
    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth"))

    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"))

    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth"))

    model = FontDiffuserModelDPM(
        unet=unet, style_encoder=style_encoder, content_encoder=content_encoder
    )
    model.to(args.device)
    print("Loaded the model state_dict successfully!")

    # Load the training ddpm_scheduler
    train_scheduler = build_ddpm_scheduler(args=args)
    print("Loaded training DDPM scheduler successfully!")

    # Load the DPM_Solver pipeline
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )
    print("Loaded DPM-solver pipeline successfully!")

    return pipe


class FontDiffuserSampler:
    """Class to handle consecutive sampling with loaded pipeline"""

    def __init__(self, args, pipe):
        self.args = args
        self.pipe = pipe
        self.device = args.device

        # Create transforms once
        self.content_transforms, self.style_transforms = create_transforms(args)

        # Load style image once
        self.style_image_pil, self.style_image_tensor = prepare_style_image(
            args, self.style_transforms
        )
        self.style_image_tensor = self.style_image_tensor.to(self.device)

        # Load TTF font once if needed
        self.ttf_font = None
        if args.character_input:
            self.ttf_font = load_ttf(ttf_path=args.ttf_path)

    def sample_single_character(self, character: str, save_dir: Optional[str] = None):
        """Generate a single character with the loaded style"""
        if save_dir is None:
            save_dir = self.args.save_image_dir

        # Set seed if specified
        if self.args.seed:
            set_seed(seed=self.args.seed)

        # Prepare content image for this character
        content_image_pil, content_image_tensor = prepare_content_image(
            self.args, self.content_transforms, character, self.ttf_font
        )

        if content_image_tensor is None:
            print(f"Skipping character '{character}' - not in font")
            return None

        # Move to device
        content_image_tensor = content_image_tensor.to(self.device)

        # Generate image
        print(f"Sampling character '{character}' with DPM-Solver++ ......")
        start = time.time()

        with torch.no_grad():
            images = self.pipe.generate(
                content_images=content_image_tensor,
                style_images=self.style_image_tensor,
                batch_size=1,
                order=self.args.order,
                num_inference_step=self.args.num_inference_steps,
                content_encoder_downsample_size=self.args.content_encoder_downsample_size,
                t_start=self.args.t_start,
                t_end=self.args.t_end,
                dm_size=self.args.content_image_size,
                algorithm_type=self.args.algorithm_type,
                skip_type=self.args.skip_type,
                method=self.args.method,
                correcting_x0_fn=self.args.correcting_x0_fn,
            )

        end = time.time()

        # Save if requested
        if self.args.save_image:
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

                # Save generated image
                char_save_dir = os.path.join(save_dir, character)
                os.makedirs(char_save_dir, exist_ok=True)

                save_single_image(save_dir=char_save_dir, image=images[0])

                # Save with content and style references
                if self.args.character_input:
                    save_image_with_content_style(
                        save_dir=char_save_dir,
                        image=images[0],
                        content_image_pil=content_image_pil,
                        content_image_path=None,
                        style_image_path=self.args.style_image_path,
                        resolution=self.args.resolution,
                    )
                else:
                    save_image_with_content_style(
                        save_dir=char_save_dir,
                        image=images[0],
                        content_image_pil=None,
                        content_image_path=self.args.content_image_path,
                        style_image_path=self.args.style_image_path,
                        resolution=self.args.resolution,
                    )

            print(f"Generated character '{character}' in {end - start:.2f}s")

        return images[0]

    def sample_multiple_characters(
        self, characters: List[str], save_dir: Optional[str] = None
    ):
        """Generate multiple characters consecutively"""
        results = {}

        for char in characters:
            try:
                generated_image = self.sample_single_character(char, save_dir)
                if generated_image is not None:
                    results[char] = generated_image
            except Exception as e:
                print(f"Error generating character '{char}': {e}")
                results[char] = None

        return results


def main():
    """Main function with batch character generation"""
    args = arg_parse()

    # Parse character list
    characters = []
    if args.content_characters:
        # Support comma-separated: "A,B,C,D" or string: "ABCD"
        if "," in args.content_characters:
            characters = [c.strip() for c in args.content_characters.split(",")]
        else:
            characters = list(args.content_characters)
    elif args.content_character:
        characters = [args.content_character]
    else:
        print(
            "No characters specified. Use --content_characters or --content_character"
        )
        return

    print(f"Will generate {len(characters)} characters: {characters}")

    # Load pipeline once
    print("Loading FontDiffuser pipeline...")
    pipe = load_fontdiffuser_pipeline(args=args)

    # Create sampler
    sampler = FontDiffuserSampler(args, pipe)

    # Generate all characters
    print(f"\nGenerating {len(characters)} characters...")
    results = sampler.sample_multiple_characters(characters, args.save_image_dir)

    # Summary
    successful = sum(1 for v in results.values() if v is not None)
    print(
        f"\nGeneration complete: {successful}/{len(characters)} characters generated successfully"
    )

    return results


if __name__ == "__main__":
    main()

"""Example
python your_script.py \
    --ckpt_dir ./checkpoints \
    --style_image_path ./style/A.png \
    --content_characters "A,B,C,D,E,F,G" \
    --save_image \
    --save_image_dir ./outputs/batch_generation \
    --character_input \
    --ttf_path ./fonts/default.ttf
"""
