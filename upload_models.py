"""
Convert PyTorch .pth weights to SafeTensors format and upload to Hugging Face Hub
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from safetensors.torch import save_file
from huggingface_hub import HfApi, create_repo, login

from utilities import load_model_checkpoint, save_model_checkpoint, find_checkpoint

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PyTorch .pth weights to SafeTensors format and upload to HF Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python upload_models.py --weights_dir "ckpt" --repo_id "username/font-diffusion-weights" --token "hf_xxx"
  python upload_models.py --weights_dir "ckpt" --repo_id "username/font-diffusion-weights" --files "unet.pth" "style_encoder.pth"
  python upload_models.py --weights_dir "ckpt" --no-upload
  python upload_models.py --weights_dir "ckpt" --repo_id "username/font-diffusion-weights" --skip-conversion
        """,
    )
    parser.add_argument("--weights_dir", type=str, required=True, default="outputs/FontDiffuser", help="Directory with .pth/.safetensors files")
    parser.add_argument("--repo_id", type=str, default="dzungpham/font-diffusion-weights", help="Hugging Face repo ID")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token")
    parser.add_argument("--files", nargs="+", default=[
        "content_encoder.pth", "content_encoder.safetensors",
        "style_encoder.pth", "style_encoder.safetensors",
        "unet.pth", "unet.safetensors",
        "total_model.pth", "total_model.safetensors",
        "scr.pth", "scr.safetensors",
    ], help="Specific files to convert (default: all standard FontDiffusion weights)")
    parser.add_argument("--repo_type", type=str, default="model", help="Repository type (default: model)")
    parser.add_argument("--private", action="store_true", default=False, help="Make repository private")
    parser.add_argument("--no-upload", action="store_true", default=False, help="Convert only, do not upload")
    parser.add_argument("--skip-conversion", action="store_true", default=False, help="Skip conversion, only upload")
    parser.add_argument("--commit-message", type=str, default="Add converted safetensors and original pth weights")
    parser.add_argument("--verbose", action="store_true", default=False, help="Verbose output")
    return parser.parse_args()

def get_token(token_arg: Optional[str]) -> Optional[str]:
    if token_arg:
        return token_arg
    token_env = os.getenv("HF_TOKEN")
    if token_env:
        logging.info("Using HF token from environment variable")
        return token_env
    try:
        from huggingface_hub import HfFolder
        return HfFolder.get_token()
    except Exception:
        return None

def validate_inputs(args: argparse.Namespace) -> bool:
    print("\n" + "=" * 70)
    print("VALIDATING INPUTS")
    print("=" * 70)
    if not os.path.isdir(args.weights_dir):
        print(f"✗ Error: Weights directory not found: {args.weights_dir}")
        return False
    print(f"✓ Weights directory: {args.weights_dir}")
    if not args.no_upload and not args.repo_id:
        print("✗ Error: --repo_id is required when uploading (use --no-upload to skip)")
        return False
    if not args.no_upload:
        token = get_token(args.token)
        if not token:
            print("✗ Error: HF token not found!")
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

def convert_pth_to_safetensors(args: argparse.Namespace) -> bool:
    if args.skip_conversion:
        print("\n⊘ Skipping conversion (--skip-conversion)")
        return True
    print("\n" + "=" * 70)
    print("CONVERTING .pth TO .safetensors")
    print("=" * 70)
    converted_count = 0
    failed_count = 0
    for file_name in args.files:
        if not file_name.endswith(".pth"):
            continue
        pth_path = os.path.join(args.weights_dir, file_name)
        safe_path = pth_path.replace(".pth", ".safetensors")
        if not os.path.isfile(pth_path):
            print(f"\n⚠ {file_name}: Not found, skipping")
            continue
        try:
            print(f"\n📦 {file_name}")
            state_dict = load_model_checkpoint(pth_path)
            if not isinstance(state_dict, dict):
                print(f"  ✗ Error: Expected dict, got {type(state_dict)}")
                failed_count += 1
                continue
            save_model_checkpoint(state_dict, safe_path)
            safe_size_mb = os.path.getsize(safe_path) / (1024 * 1024)
            pth_size_mb = os.path.getsize(pth_path) / (1024 * 1024)
            compression = ((pth_size_mb - safe_size_mb) / pth_size_mb) * 100 if pth_size_mb else 0
            print(f"  ✓ Saved to: {safe_path}")
            print(f"    Original: {pth_size_mb:.2f} MB → Safetensors: {safe_size_mb:.2f} MB")
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
        token = get_token(args.token)
        if not token:
            print("✗ Error: HF token not found")
            return False
        login(token=token)
        api = HfApi()
        print(f"\n📤 Creating/verifying repository...")
        create_repo(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            exist_ok=True,
            private=args.private,
            token=token,
        )
        print(f"✓ Repository ready\n")
        print(f"📤 Uploading folder: {args.weights_dir}")
        api.upload_folder(
            folder_path=args.weights_dir,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
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