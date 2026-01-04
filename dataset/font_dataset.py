import os
import random
from PIL import Image
from typing import Optional, Dict, Any, List, Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def get_nonorm_transform(resolution):
    nonorm_transform = transforms.Compose(
        [
            transforms.Resize(
                (resolution, resolution),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ]
    )
    return nonorm_transform


class FontDataset(Dataset):
    """The dataset of font generation with optional style transformation support"""

    def __init__(
        self,
        args,
        phase: str,
        transforms: Optional[List] = None,
        scr: bool = False,
        include_source_style: bool = False,  # ✅ ADD THIS
    ):
        super().__init__()
        self.root = args.data_root
        self.phase = phase
        self.scr = scr
        self.include_source_style = include_source_style  # ✅ ADD THIS

        if self.scr:
            self.num_neg = args.num_neg

        # Get Data path
        self.get_path()
        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)

    def get_path(self):
        """Build mapping of target images to styles and content."""
        self.target_images: List[str] = []
        
        # images with related style: {style: [img_path1, img_path2, ...]}
        self.style_to_images: Dict[str, List[str]] = {}
        
        # ✅ ADD: Mapping of content to available styles
        # {content: {style: [img_path1, img_path2, ...]}}
        self.content_to_styles: Dict[str, Dict[str, List[str]]] = {}
        
        # ✅ ADD: Mapping of style to content
        # {style: [content1, content2, ...]}
        self.style_to_content: Dict[str, List[str]] = {}
        
        target_image_dir = f"{self.root}/{self.phase}/TargetImage"
        
        for style in os.listdir(target_image_dir):
            style_path = os.path.join(target_image_dir, style)
            
            if not os.path.isdir(style_path):
                continue
            
            images_related_style: List[str] = []
            contents_in_style: List[str] = []
            
            for img in os.listdir(style_path):
                img_path = os.path.join(style_path, img)
                self.target_images.append(img_path)
                images_related_style.append(img_path)
                
                # Extract content from filename
                # Filename format: {style}+{content_hash}.png or {style}+{char}.png
                try:
                    img_name = img.split(".")[0]  # Remove .png
                    parts = img_name.split("+")
                    
                    if len(parts) >= 2:
                        content = parts[1]  # content_hash or character
                        
                        if content not in contents_in_style:
                            contents_in_style.append(content)
                        
                        # Build content_to_styles mapping
                        if content not in self.content_to_styles:
                            self.content_to_styles[content] = {}
                        
                        if style not in self.content_to_styles[content]:
                            self.content_to_styles[content][style] = []
                        
                        self.content_to_styles[content][style].append(img_path)
                
                except Exception as e:
                    print(f"Warning: Could not parse image name {img}: {e}")
                    continue
            
            self.style_to_images[style] = images_related_style
            self.style_to_content[style] = contents_in_style

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get dataset item with optional source style for style transformation."""
        target_image_path = self.target_images[index]
        target_image_name = target_image_path.split("/")[-1]
        
        # Parse filename: style+content.png
        try:
            img_name_without_ext = target_image_name.split(".")[0]
            parts = img_name_without_ext.split("+")
            style = parts[0]
            content = parts[1] if len(parts) > 1 else ""
        except Exception as e:
            raise ValueError(f"Invalid image name format {target_image_name}: {e}")

        # Read content image
        content_image_path = f"{self.root}/{self.phase}/ContentImage/{content}.png"
        content_image = Image.open(content_image_path).convert("RGB")

        # Random sample for style image (reference style, same content but different style)
        images_related_style = self.style_to_images[style].copy()
        if target_image_path in images_related_style:
            images_related_style.remove(target_image_path)
        
        if not images_related_style:
            raise ValueError(
                f"No other images found for style {style} to sample from"
            )
        
        style_image_path = random.choice(images_related_style)
        style_image = Image.open(style_image_path).convert("RGB")

        # Read target image
        target_image = Image.open(target_image_path).convert("RGB")
        nonorm_target_image = self.nonorm_transforms(target_image)

        # Apply transforms
        if self.transforms is not None:
            content_image = self.transforms[0](content_image)
            style_image = self.transforms[1](style_image)
            target_image = self.transforms[2](target_image)

        sample: Dict[str, Any] = {
            "content_image": content_image,
            "style_image": style_image,
            "target_image": target_image,
            "target_image_path": target_image_path,
            "nonorm_target_image": nonorm_target_image,
        }

        # ✅ ADD SOURCE STYLE IMAGE FOR STYLE TRANSFORMATION MODULE
        if self.include_source_style:
            source_style_image = self._get_source_style_image(
                content, style, images_related_style
            )
            if source_style_image is not None:
                sample["source_style_image"] = source_style_image
            else:
                # Fallback: use a different random style as source
                sample["source_style_image"] = style_image

        # Add negative images for SCR loss
        if self.scr:
            neg_images = self._get_negative_images(content, style)
            sample["neg_images"] = neg_images

        return sample

    def _get_source_style_image(
        self,
        content: str,
        target_style: str,
        exclude_paths: List[str],
    ) -> Optional[torch.Tensor]:
        """
        Get source style image for style transformation.
        
        Selects a different style with the same content as source style.
        This creates a (source_style -> target_style) transformation pair.
        
        Args:
            content: Content identifier
            target_style: Current target style
            exclude_paths: Paths to exclude from selection
        
        Returns:
            Transformed source style image or None
        """
        if content not in self.content_to_styles:
            return None
        
        # Get all available styles for this content
        available_styles = list(self.content_to_styles[content].keys())
        
        # Remove target style to get a different source style
        if target_style in available_styles:
            available_styles.remove(target_style)
        
        if not available_styles:
            return None
        
        # Randomly select a source style
        source_style = random.choice(available_styles)
        
        # Get an image from the source style for this content
        source_images = self.content_to_styles[content][source_style]
        source_image_path = random.choice(source_images)
        
        try:
            source_style_image = Image.open(source_image_path).convert("RGB")
            
            # Apply style transform
            if self.transforms is not None:
                source_style_image = self.transforms[1](source_style_image)
            
            return source_style_image
        
        except Exception as e:
            print(
                f"Warning: Could not load source style image from {source_image_path}: {e}"
            )
            return None

    def _get_negative_images(self, content: str, style: str) -> torch.Tensor:
        """
        Get negative images from different styles of the same content.
        Used for SCR (Style Content Recognition) loss.
        
        Args:
            content: Content identifier
            style: Current style
        
        Returns:
            Tensor of negative images (num_neg, C, H, W)
        """
        neg_images = None
        
        if content in self.content_to_styles:
            # Get all styles for this content except current style
            available_styles = [
                s for s in self.content_to_styles[content].keys() if s != style
            ]
            
            # Limit to num_neg styles
            num_neg = min(self.num_neg, len(available_styles))
            
            if num_neg > 0:
                selected_styles = random.sample(available_styles, num_neg)
                
                for style_idx, neg_style in enumerate(selected_styles):
                    neg_image_paths = self.content_to_styles[content][neg_style]
                    neg_image_path = random.choice(neg_image_paths)
                    
                    try:
                        neg_image = Image.open(neg_image_path).convert("RGB")
                        
                        if self.transforms is not None:
                            neg_image = self.transforms[2](neg_image)
                        
                        if neg_images is None:
                            neg_images = neg_image[None, :, :, :]
                        else:
                            neg_images = torch.cat(
                                [neg_images, neg_image[None, :, :, :]], dim=0
                            )
                    
                    except Exception as e:
                        print(
                            f"Warning: Could not load negative image from {neg_image_path}: {e}"
                        )
                        continue
        
        # Fallback: if no negative images found, create dummy tensor
        if neg_images is None:
            num_neg = getattr(self, 'num_neg', 1)
            # Use a black image as fallback
            neg_images = torch.zeros(
                num_neg, 3, 256, 256, dtype=torch.float32
            )
        
        return neg_images

    def __len__(self) -> int:
        return len(self.target_images)


# ✅ Optional: Utility class for debugging dataset structure
class FontDatasetInspector:
    """Inspect and validate FontDataset structure."""
    
    @staticmethod
    def inspect(dataset: FontDataset) -> Dict[str, Any]:
        """Inspect dataset structure and statistics."""
        return {
            "num_samples": len(dataset),
            "num_styles": len(dataset.style_to_images),
            "styles": list(dataset.style_to_images.keys()),
            "num_contents": len(dataset.content_to_styles),
            "content_style_pairs": {
                content: list(styles.keys())
                for content, styles in dataset.content_to_styles.items()
            },
            "style_content_pairs": dataset.style_to_content,
        }
    
    @staticmethod
    def validate(dataset: FontDataset) -> Tuple[bool, List[str]]:
        """
        Validate dataset integrity.
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check if all target images exist
        for img_path in dataset.target_images:
            if not os.path.exists(img_path):
                errors.append(f"Target image not found: {img_path}")
        
        # Check if content images exist
        for content in dataset.content_to_styles.keys():
            content_path = f"{dataset.root}/{dataset.phase}/ContentImage/{content}.png"
            if not os.path.exists(content_path):
                errors.append(f"Content image not found: {content_path}")
        
        # Check if each content has at least 2 styles for style transformation
        if dataset.include_source_style:
            for content, styles in dataset.content_to_styles.items():
                if len(styles) < 2:
                    errors.append(
                        f"Content '{content}' has only 1 style, need at least 2 for style transformation"
                    )
        
        is_valid = len(errors) == 0
        return is_valid, errors