"""
Robust Font Manager for FontDiffuser Batch Processing
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import defaultdict
from functools import lru_cache

import numpy as np
from PIL import Image
import pygame
import pygame.freetype
from fontTools.ttLib import TTFont
import cv2


def ttf2im_robust(font, char, canvas_size=256):
    """
    Robust font rendering with proper centering and edge protection.

    Args:
        font: pygame.freetype.Font object
        char: Character to render
        canvas_size: Size of output square image

    Returns:
        PIL Image of the rendered character, or None on failure
    """
    try:
        # Try to render the character
        surface, _ = font.render(char, size=canvas_size)
    except Exception as e:
        print(f"Failed to render character '{char}': {e}")
        return None

    try:
        # Create white background
        bg = np.full((canvas_size, canvas_size), 255, dtype=np.uint8)

        # Get alpha channel and invert (black character on transparent background)
        imo = pygame.surfarray.pixels_alpha(surface).transpose(1, 0)
        imo = 255 - np.array(Image.fromarray(imo))

        # Make a copy for modification
        result = bg.copy()

        # Get original dimensions
        h, w = imo.shape[:2]

        # Resize only if character exceeds canvas - with safety
        if h > canvas_size or w > canvas_size:
            # Calculate scaling factor to fit within canvas
            scale = min(canvas_size / h, canvas_size / w)

            # Ensure minimum dimensions of 1 pixel
            new_h = max(1, int(h * scale))
            new_w = max(1, int(w * scale))

            # Resize with area interpolation (good for downscaling)
            imo = cv2.resize(imo, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w  # Update dimensions

        # Calculate centering position with bounds checking
        x = (canvas_size - w) // 2
        y = (canvas_size - h) // 2

        # Ensure positions are within valid range
        x = max(0, min(x, canvas_size - w))
        y = max(0, min(y, canvas_size - h))

        # Calculate end positions
        end_x = x + w
        end_y = y + h

        # Final bounds check
        if (
            x >= 0
            and y >= 0
            and end_x <= canvas_size
            and end_y <= canvas_size
            and w > 0
            and h > 0
        ):
            # CORRECT placement: im[y:y+h, x:x+w] = imo
            result[y:end_y, x:end_x] = imo
        else:
            # Fallback: Use the entire canvas if placement fails
            print(f"Warning: Placement error for '{char}', using center crop")
            if w <= canvas_size and h <= canvas_size:
                # Simple center placement
                center_x = (canvas_size - w) // 2
                center_y = (canvas_size - h) // 2
                result[center_y : center_y + h, center_x : center_x + w] = imo

        # Convert to PIL Image
        pil_image = Image.fromarray(result.astype("uint8")).convert("RGB")

        # Optional: Add a visual debug border for problematic characters
        if hasattr(font, "_debug") and font._debug:
            from PIL import ImageDraw

            draw = ImageDraw.Draw(pil_image)
            draw.rectangle(
                [0, 0, canvas_size - 1, canvas_size - 1], outline="red", width=1
            )

        return pil_image

    except Exception as e:
        print(f"Error processing character '{char}': {e}")
        return None


class FontRenderer:
    """
    Robust font renderer with fixed ttf2im and additional safety checks
    """

    def __init__(self, font_size: int = 256, canvas_size: int = 256):
        """
        Args:
            font_size: Font size for rendering
            canvas_size: Size of output image (should be >= font_size)
        """
        self.font_size = font_size
        self.canvas_size = canvas_size

        # Initialize pygame for font rendering
        if not pygame.get_init():
            pygame.init()

    def ttf2im_fixed(
        self, font, char: str, debug: bool = False
    ) -> Optional[Image.Image]:
        """
        Fixed version of ttf2im that properly centers characters without cropping

        Args:
            font: pygame.freetype.Font object
            char: Character to render
            debug: Whether to save debug images

        Returns:
            PIL Image of rendered character or None on failure
        """
        return ttf2im_robust(font, char, self.canvas_size)

    def _create_default_x_marker(self, char: str) -> Image.Image:
        """Create a default X marker for invalid/unrenderable characters"""
        img = np.full((self.canvas_size, self.canvas_size, 3), 255, dtype=np.uint8)

        # Draw a red X
        cv2.line(
            img,
            (50, 50),
            (self.canvas_size - 50, self.canvas_size - 50),
            (255, 0, 0),
            5,
        )
        cv2.line(
            img,
            (self.canvas_size - 50, 50),
            (50, self.canvas_size - 50),
            (255, 0, 0),
            5,
        )

        # Add character label
        cv2.putText(
            img,
            char,
            (self.canvas_size // 2 - 20, self.canvas_size // 2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )

        return Image.fromarray(img)

    def render_with_safety_check(self, font, char: str) -> Optional[Image.Image]:
        """
        Render character with additional safety checks for edge cases
        """
        result = self.ttf2im_fixed(font, char, debug=False)

        if result is None:
            return None

        # Convert to numpy for analysis
        img_array = np.array(result.convert("L"))  # Convert to grayscale

        # Check for edge artifacts (character touching edges)
        edge_thickness = 2
        top_edge = img_array[:edge_thickness, :]
        bottom_edge = img_array[-edge_thickness:, :]
        left_edge = img_array[:, :edge_thickness]
        right_edge = img_array[:, -edge_thickness:]

        # Check if character is too close to edges
        edge_threshold = 200  # Below this value indicates character pixel
        edges_touching = [
            np.any(top_edge < edge_threshold),
            np.any(bottom_edge < edge_threshold),
            np.any(left_edge < edge_threshold),
            np.any(right_edge < edge_threshold),
        ]

        if any(edges_touching):
            # Character is touching edges, render with more padding
            print(f"  Warning: Character '{char}' touches edges, adjusting...")
            # Re-render with larger canvas
            original_size = self.canvas_size
            self.canvas_size = int(original_size * 1.5)
            result = self.ttf2im_fixed(font, char, debug=False)
            self.canvas_size = original_size

            # Resize back to original size
            if result:
                result = result.resize((original_size, original_size), Image.LANCZOS)

        return result


class FontManager:
    """
    Efficient font manager with robust rendering and Unicode character support checking
    """

    def __init__(
        self, font_paths: List[str], font_size: int = 256, canvas_size: int = 256
    ):
        """
        Initialize font manager with multiple font paths

        Args:
            font_paths: List of paths to TTF font files
            font_size: Default font size for rendering
            canvas_size: Size of output image
        """
        self.font_paths = font_paths
        self.font_size = font_size
        self.canvas_size = canvas_size

        self.available_fonts = []
        self.unicode_coverage_cache = {}
        self.renderer = FontRenderer(font_size, canvas_size)

        self._load_fonts()

    def _load_fonts(self):
        """Load and validate all fonts"""
        print(f"\nLoading {len(self.font_paths)} font(s)...")

        for font_path in self.font_paths:
            try:
                if not os.path.exists(font_path):
                    print(f"  ✗ Font file not found: {font_path}")
                    continue

                # Load with pygame for rendering
                font = pygame.freetype.Font(font_path, size=self.font_size)

                # Load with fontTools for Unicode coverage checking
                ttfont = TTFont(font_path)

                # Get font name for display
                try:
                    name_record = ttfont["name"].getName(4, 3, 1)  # Full font name
                    font_name = (
                        name_record.toUnicode() if name_record else Path(font_path).stem
                    )
                except:
                    font_name = Path(font_path).stem

                # Test font rendering with a sample character
                test_char = (
                    "A"
                    if self._has_character(ttfont, ord("A"))
                    else next(iter(self._get_unicode_coverage(ttfont)), "?")
                )

                if test_char:
                    test_image = self.renderer.ttf2im_fixed(
                        font, test_char, debug=False
                    )
                    if test_image is None:
                        print(f"  ✗ Font '{font_name}' failed rendering test")
                        continue

                self.available_fonts.append(
                    {
                        "path": font_path,
                        "name": font_name,
                        "pygame_font": font,
                        "ttfont": ttfont,
                        "unicode_coverage": self._get_unicode_coverage(ttfont),
                    }
                )

                print(f"  ✓ Loaded: {font_name}")

            except Exception as e:
                print(f"  ✗ Failed to load font {font_path}: {e}")

        if not self.available_fonts:
            raise ValueError("No valid fonts were loaded")

        print(f"Successfully loaded {len(self.available_fonts)} font(s)")

    @lru_cache(maxsize=128)
    def _get_unicode_coverage(self, ttfont: TTFont) -> Set[int]:
        """Extract all supported Unicode code points from a font"""
        coverage = set()
        if "cmap" in ttfont:
            cmap = ttfont["cmap"]
            for subtable in cmap.tables:
                coverage.update(subtable.cmap.keys())
        return coverage

    def _has_character(self, ttfont: TTFont, char_code: int) -> bool:
        """Check if a font has a specific character"""
        if "cmap" not in ttfont:
            return False

        cmap = ttfont["cmap"]
        for subtable in cmap.tables:
            if char_code in subtable.cmap:
                return True
        return False

    @lru_cache(maxsize=1024)
    def get_supporting_font(self, char: str) -> Optional[Dict]:
        """
        Find the first font that supports the given character

        Args:
            char: Single character string

        Returns:
            Font dictionary or None if no font supports the character
        """
        char_code = ord(char)

        for font_info in self.available_fonts:
            if char_code in font_info["unicode_coverage"]:
                return font_info

        return None

    def find_fonts_for_characters(
        self, characters: List[str]
    ) -> Dict[str, Optional[str]]:
        """
        Find which font supports each character in a list

        Args:
            characters: List of characters to check

        Returns:
            Dictionary mapping character to font name (or None if no font supports it)
        """
        font_mapping = {}

        for char in characters:
            font_info = self.get_supporting_font(char)
            if font_info:
                font_mapping[char] = font_info["name"]
            else:
                font_mapping[char] = None

        return font_mapping

    def can_render_character(self, char: str) -> bool:
        """Check if any loaded font can render the character"""
        return self.get_supporting_font(char) is not None

    @lru_cache(maxsize=1024)
    def render_character(self, char: str) -> Optional[Image.Image]:
        """
        Render character using the first available supporting font

        Args:
            char: Single character string

        Returns:
            PIL Image or None if character cannot be rendered
        """
        font_info = self.get_supporting_font(char)
        if not font_info:
            print(f"    No font supports character: '{char}' (U+{ord(char):04X})")
            return None

        try:
            # Render with safety checks
            result = self.renderer.render_with_safety_check(
                font_info["pygame_font"], char
            )

            if result is None:
                print(f"    Failed to render character: '{char}'")
                return None

            # Additional quality check
            img_array = np.array(result.convert("L"))

            # Check if image is mostly empty
            if np.std(img_array) < 5:  # Very low contrast
                print(f"    Warning: Character '{char}' rendered as mostly blank")
                # Try alternative rendering approach
                result = self._render_alternative(char, font_info["pygame_font"])

            return result

        except Exception as e:
            print(f"    Error rendering character '{char}': {e}")
            return None

    def _render_alternative(self, char: str, font) -> Image.Image:
        """Alternative rendering method for difficult characters"""
        # Create larger surface
        surface_size = self.canvas_size * 2
        surface = pygame.Surface((surface_size, surface_size), pygame.SRCALPHA)
        surface.fill((255, 255, 255, 255))  # White background

        # Render character
        text_surface, text_rect = font.render(
            char,
            fgcolor=(0, 0, 0),  # Black
            size=self.font_size * 2,  # Larger size
        )

        # Center
        text_rect.center = (surface_size // 2, surface_size // 2)
        surface.blit(text_surface, text_rect)

        # Convert to PIL and resize
        img_str = pygame.image.tostring(surface, "RGBA")
        pil_image = Image.frombytes("RGBA", (surface_size, surface_size), img_str)

        # Convert to RGB and resize
        pil_image = pil_image.convert("RGB").resize(
            (self.canvas_size, self.canvas_size), Image.LANCZOS
        )

        return pil_image

    def batch_render_characters(
        self, characters: List[str]
    ) -> Dict[str, Optional[Image.Image]]:
        """
        Render multiple characters efficiently

        Returns:
            Dictionary mapping character to rendered image
        """
        results = {}

        for char in characters:
            results[char] = self.render_character(char)

        return results

    def get_font_statistics(self) -> Dict:
        """Get statistics about loaded fonts"""
        stats = {
            "total_fonts": len(self.available_fonts),
            "font_names": [f["name"] for f in self.available_fonts],
            "font_paths": [f["path"] for f in self.available_fonts],
            "total_glyphs_by_font": {},
            "sample_coverage": {},
        }

        # Test coverage with common characters
        test_sets = {
            "basic_latin": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            "common_chinese": "的一是不了人我在有他这为之大来以个中上们",
            "punctuation": "，。？！；：''（）【】《》",
        }

        for font_info in self.available_fonts:
            font_name = font_info["name"]
            coverage = font_info["unicode_coverage"]

            stats["total_glyphs_by_font"][font_name] = len(coverage)

            # Calculate coverage for each test set
            coverage_stats = {}
            for set_name, test_chars in test_sets.items():
                supported = sum(1 for c in test_chars if ord(c) in coverage)
                total = len(test_chars)
                coverage_pct = (supported / total * 100) if total > 0 else 0
                coverage_stats[set_name] = {
                    "supported": supported,
                    "total": total,
                    "percentage": round(coverage_pct, 1),
                }

            stats["sample_coverage"][font_name] = coverage_stats

        return stats
