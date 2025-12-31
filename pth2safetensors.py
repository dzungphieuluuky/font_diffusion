"""
Convert PyTorch .pth weights to SafeTensors format and upload to Hugging Face Hub
Supports command-line arguments for flexibility
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any
from safetensors.torch import save_file
from huggingface_hub import HfApi, create_repo, login


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Convert PyTorch .pth weights to SafeTensors format and upload to HF Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with all arguments
  python pth2safetensors.py \\
    --weights_dir "ckpt" \\
    --repo_id "username/font-diffusion-weights" \\
    --token "hf_xxxxxxxxxxxxx"

  # Use environment variable for token
  export HF_TOKEN="hf_xxxxxxxxxxxxx"
  python pth2safetensors.py \\
    --weights_dir "ckpt" \\
    --repo_id "username/font-diffusion-weights"

  # Convert only specific files
  python pth2safetensors.py \\
    --weights_dir "ckpt" \\
    --repo_id "username/font-diffusion-weights" \\
    --files "unet.pth" "style_encoder.pth"

  # Convert without uploading
  python pth2safetensors.py \\
    --weights_dir "ckpt" \\
    --no-upload

  # Upload existing .safetensors files (no conversion)
  python pth2safetensors.py \\
    --weights_dir "ckpt" \\
    --repo_id "username/font-diffusion-weights" \\
    --skip-conversion
        """,
    )

    parser.add_argument(
        "--weights_dir",
        type=str,
        required=True,
        help="Path to directory containing .pth files (default: ckpt)",
    )

    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Hugging Face repo ID (e.g., username/font-diffusion-weights). Required if uploading.",
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token. If not provided, uses HF_TOKEN environment variable.",
    )

    parser.add_argument(
        "--files",
        nargs="+",
        default=[
            "content_encoder.pth",
            "style_encoder.pth",
            "unet.pth",
            "total_model.pth",
            "scr.pth",
        ],
        help="Specific .pth files to convert (default: all standard FontDiffusion weights)",
    )

    parser.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Make repository private (default: True)",
    )

    parser.add_argument(
        "--public",
        action="store_true",
        default=False,
        help="Make repository public (overrides --private)",
    )

    parser.add_argument(
        "--no-upload",
        action="store_true",
        default=False,
        help="Convert to safetensors but do not upload to Hub",
    )

    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        default=False,
        help="Skip conversion, only upload existing .safetensors files",
    )

    parser.add_argument(
        "--commit-message",
        type=str,
        default="Add converted safetensors and original pth weights",
        help="Custom commit message for Hub upload",
    )

    parser.add_argument(
        "--weights_only",
        action="store_true",
        default=True,
        help="Use weights_only=True when loading .pth (safer but stricter)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print detailed information during conversion",
    )

    return parser.parse_args()


def get_token(token_arg: Optional[str]) -> Optional[str]:
    """Get HF token from argument or environment variable"""
    if token_arg:
        return token_arg

    token_env = os.getenv("HF_TOKEN")
    if token_env:
        return token_env

    try:
        from huggingface_hub import HfFolder

        return HfFolder.get_token()
    except Exception:
        pass

    return None


def validate_inputs(args: argparse.Namespace) -> bool:
    """Validate command-line arguments"""
    print("\n" + "=" * 70)
    print("VALIDATING INPUTS")
    print("=" * 70)

    if not os.path.isdir(args.weights_dir):
        print(f"✗ Error: Weights directory not found: {args.weights_dir}")
        return False

    print(f"✓ Weights directory: {args.weights_dir}")
    print(f"  Contents: {len(os.listdir(args.weights_dir))} files")

    if not args.no_upload:
        if not args.repo_id:
            print(
                "✗ Error: --repo_id is required when uploading (use --no-upload to skip)"
            )
            return False

        print(f"✓ Repository ID: {args.repo_id}")

        token = get_token(args.token)
        if not token:
            print("✗ Error: HF token not found!")
            print("  Provide via:")
            print("    1. --token argument")
            print("    2. HF_TOKEN environment variable")
            print("    3. huggingface-cli login")
            return False

        print(f"✓ HF token: {'*' * 20}")
    else:
        print("⊘ Skipping upload (--no-upload)")

    print(f"\n✓ Files to process: {len(args.files)}")
    for file_name in args.files:
        file_path = os.path.join(args.weights_dir, file_name)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  ✓ {file_name} ({size_mb:.2f} MB)")
        else:
            print(f"  ⚠ {file_name} (not found)")

    return True


def load_pth_file(
    pth_path: str, weights_only: bool = True, verbose: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Load a .pth file safely

    Args:
        pth_path: Path to .pth file
        weights_only: Use weights_only mode (safer)
        verbose: Print debug info

    Returns:
        State dict or None if failed
    """
    try:
        if verbose:
            print(f"  Loading: {pth_path}")

        state_dict = torch.load(pth_path, map_location="cpu", weights_only=weights_only)

        if verbose:
            print(f"  Type: {type(state_dict)}")

        # Unwrap nested state_dict
        if isinstance(state_dict, dict):
            if "state_dict" in state_dict:
                if verbose:
                    print(f"  Unwrapping 'state_dict' key...")
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                if verbose:
                    print(f"  Unwrapping 'model' key...")
                state_dict = state_dict["model"]

        return state_dict if isinstance(state_dict, dict) else None

    except Exception as e:
        print(f"  ✗ Failed to load: {e}")
        return None


def convert_pth_to_safetensors(args: argparse.Namespace) -> bool:
    """Convert .pth files to safetensors format"""
    if args.skip_conversion:
        print("\n⊘ Skipping conversion (--skip-conversion)")
        return True

    print("\n" + "=" * 70)
    print("CONVERTING .pth TO SAFETENSORS")
    print("=" * 70)

    converted_count = 0
    failed_count = 0

    for file_name in args.files:
        pth_path = os.path.join(args.weights_dir, file_name)
        safe_path = pth_path.replace(".pth", ".safetensors")

        if not os.path.isfile(pth_path):
            print(f"\n⚠ {file_name}: Not found, skipping")
            continue

        try:
            print(f"\n📦 {file_name}")

            # Load weights
            state_dict = load_pth_file(
                pth_path, weights_only=args.weights_only, verbose=args.verbose
            )

            if state_dict is None:
                print(f"  ✗ Error: Failed to load state dict")
                failed_count += 1
                continue

            if not isinstance(state_dict, dict):
                print(f"  ✗ Error: Expected dict, got {type(state_dict)}")
                failed_count += 1
                continue

            if args.verbose:
                num_params = sum(
                    v.numel() if hasattr(v, "numel") else 0 for v in state_dict.values()
                )
                print(f"  Parameters: {num_params:,}")

            # Save as safetensors
            save_file(state_dict, safe_path)

            safe_size_mb = os.path.getsize(safe_path) / (1024 * 1024)
            pth_size_mb = os.path.getsize(pth_path) / (1024 * 1024)
            compression = ((pth_size_mb - safe_size_mb) / pth_size_mb) * 100

            print(f"  ✓ Saved to: {safe_path}")
            print(
                f"    Original: {pth_size_mb:.2f} MB → Safetensors: {safe_size_mb:.2f} MB"
            )
            print(f"    Compression: {compression:.1f}%")

            converted_count += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed_count += 1
            if args.verbose:
                import traceback

                traceback.print_exc()

    print("\n" + "-" * 70)
    print(f"Conversion complete: {converted_count} succeeded, {failed_count} failed")

    return failed_count == 0


def upload_to_hub(args: argparse.Namespace) -> bool:
    """Upload converted weights to Hugging Face Hub"""
    if args.no_upload:
        print("\n⊘ Skipping upload (--no-upload)")
        return True

    if not args.repo_id:
        print("\n✗ Error: --repo_id required for upload")
        return False

    print("\n" + "=" * 70)
    print("UPLOADING TO HUGGING FACE HUB")
    print("=" * 70)

    try:
        # Get token
        token = get_token(args.token)
        if not token:
            print("✗ Error: HF token not found")
            return False

        # Login
        login(token=token)
        api = HfApi()

        # Determine privacy
        private = args.private and not args.public

        print(f"\n📤 Creating/verifying repository...")
        print(f"  Repo ID: {args.repo_id}")
        print(f"  Private: {private}")

        create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            exist_ok=True,
            private=private,
            token=token,
        )

        print(f"✓ Repository ready\n")

        # Upload folder
        print(f"📤 Uploading folder: {args.weights_dir}")
        api.upload_folder(
            folder_path=args.weights_dir,
            repo_id=args.repo_id,
            repo_type="model",
            token=token,
            commit_message=args.commit_message,
        )

        model_url = f"https://huggingface.co/models/{args.repo_id}"
        print(f"\n✓ Upload successful!")
        print(f"  Repository URL: {model_url}")

        return True

    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return False


def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("PYTORCH TO SAFETENSORS CONVERTER & HF UPLOADER")
    print("=" * 70)

    args = parse_arguments()

    if args.verbose:
        print("\n📋 Arguments:")
        for key, value in vars(args).items():
            if key == "token" and value:
                print(f"  {key}: {'*' * 20}")
            else:
                print(f"  {key}: {value}")

    if not validate_inputs(args):
        print("\n✗ Validation failed")
        sys.exit(1)

    if not convert_pth_to_safetensors(args):
        print("\n✗ Conversion failed")
        if not args.skip_conversion:
            sys.exit(1)

    if not upload_to_hub(args):
        print("\n✗ Upload failed")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("✓ ALL DONE!")
    print("=" * 70)

    if not args.no_upload:
        print(f"\n📦 Your weights are now available at:")
        print(f"   https://huggingface.co/models/{args.repo_id}")
        print(f"\n📖 Load them with:")
        print(f"   from safetensors.torch import load_file")
        print(f"   state = load_file('model.safetensors')")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

"""Example
Examples:
  # Basic usage with all arguments
  python pth2safetensors.py \
    --weights_dir "ckpt" \
    --repo_id "username/font-diffusion-weights" \
    --token "hf_xxxxxxxxxxxxx"

  # Use environment variable for token
  export HF_TOKEN="hf_xxxxxxxxxxxxx"
  python pth2safetensors.py \
    --weights_dir "ckpt" \
    --repo_id "username/font-diffusion-weights"

  # Convert only specific files
  python pth2safetensors.py \
    --weights_dir "ckpt" \
    --repo_id "username/font-diffusion-weights" \
    --files "unet.pth" "style_encoder.pth"

  # Convert without uploading
  python pth2safetensors.py \
    --weights_dir "ckpt" \
    --no-upload

  # Upload existing .safetensors files (no conversion)
  python pth2safetensors.py \
    --weights_dir "ckpt" \
    --repo_id "username/font-diffusion-weights" \
    --skip-conversion
"""
