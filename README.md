# FontDiffuser

FontDiffuser is a neural network-based font style transfer and generation toolkit. It leverages diffusion models to synthesize or transfer font styles, supporting both single and batch character processing with a variety of optimizations for efficient inference.

## Features

- **Font Style Transfer:** Generate new font images by transferring style from one font/image to another.
- **Optimized Inference:** Supports FP16, torch.compile, channels_last, xformers, and other optimizations for fast and memory-efficient inference.
- **Batch Processing:** Efficiently process multiple characters in a single run.
- **Caching:** Uses LRU caching for font loading, character checks, and image transforms.
- **Flexible Input:** Accepts both character and image inputs for content and style.
- **Safe for Pretrained Weights:** All optimizations are safe and do not alter model weights or outputs.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dzungphieuluuky/FontDiffusion.git
   cd FontDiffusion
   ```

2. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

3. **(Optional) Install xformers for memory-efficient attention:**
   ```bash
   uv pip install xformers
   ```

## Usage

### Optimized Inference

Run the optimized sampling script:

```bash
python sample_optimized.py --ckpt_dir path/to/checkpoints --content_character "A" --style_image_path path/to/style.png --save_image --save_image_dir results/
```

**Common arguments:**
- `--ckpt_dir`: Path to pretrained model checkpoints.
- `--content_character`: Character to generate (if using font input).
- `--content_image_path`: Path to content image (if using image input).
- `--style_image_path`: Path to style image.
- `--ttf_path`: Path to TTF font file.
- `--save_image`: Save generated images.
- `--save_image_dir`: Directory to save results.
- `--device`: Device for inference (e.g., `cuda:0`).
- `--fp16`: Use half-precision inference.
- `--enable_xformers`: Enable xformers memory-efficient attention.

For a full list of options, run:
```bash
python sample_optimized.py --help
```

### Batch Processing

To process multiple characters in a batch:
```bash
python sample_optimized.py --ckpt_dir path/to/checkpoints --character_input --content_character "ABCD" --style_image_path path/to/style.png --batch_size 4 --save_image --save_image_dir results/
```

## Project Structure

```
font_diffusion/
├── configs/                # Configuration files and argument parsers
├── src/                    # Model architectures and pipeline
├── utils.py                # Utility functions (font loading, image saving, etc.)
├── sample_optimized.py     # Main optimized inference script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Model Checkpoints

Download pretrained checkpoints and place them in a directory (e.g., `checkpoints/`). Specify this path with `--ckpt_dir`.

## Citation

If you use FontDiffuser in your research, please cite:

```
@misc{fontdiffuser2025,
  title={FontDiffuser: Diffusion Models for Font Style Transfer},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/font_diffusion}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Diffusers](https://github.com/huggingface/diffusers)
- [xformers](https://github.com/facebookresearch/xformers)
- [Pillow](https://python-pillow.org/)

---
For questions or contributions, please open an issue or pull request.