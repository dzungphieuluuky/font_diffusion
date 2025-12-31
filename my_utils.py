import os
import shutil


def flatten_folder(root_dir):
    """
    Moves all files from subdirectories of root_dir into root_dir.
    """
    for subdir, dirs, files in os.walk(root_dir):
        if subdir == root_dir:
            continue  # Skip the root directory itself
        for file in files:
            src = os.path.join(subdir, file)
            dst = os.path.join(root_dir, file)
            # If a file with the same name exists, rename to avoid overwrite
            if os.path.exists(dst):
                base, ext = os.path.splitext(file)
                i = 1
                while os.path.exists(os.path.join(root_dir, f"{base}_{i}{ext}")):
                    i += 1
                dst = os.path.join(root_dir, f"{base}_{i}{ext}")
            shutil.move(src, dst)
    # Optionally, remove empty subdirectories
    for subdir, dirs, files in os.walk(root_dir, topdown=False):
        if subdir != root_dir and not os.listdir(subdir):
            os.rmdir(subdir)


if __name__ == "__main__":
    flatten_folder(r"data\content\nomtutao_output\NomNaTong-Regular")
