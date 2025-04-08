from PIL import Image
import wandb
from pathlib import Path
from tqdm import tqdm
import fnmatch
from IPython.display import Image as im, HTML
from random import sample
import cv2
import os
from base64 import b64encode

import wandb
import os
from PIL import Image
import imageio
import re


def create_gif(directory=".", duration=200, loop=0):
    """
    Creates a GIF from a series of PNG images in a directory.

    Args:
        directory (str): The directory containing the PNG images. Defaults to the current directory.
        duration (int): The duration (in milliseconds) of each frame in the GIF. Defaults to 200.
        loop (int): Number of times the GIF should loop.  0 means infinite loop. Defaults to 0 (infinite).
    """
    images = []
    filenames = [fn.split() for fn in os.listdir(directory) if fn.endswith(".png")]
    files = [filenames[i][0] for i in range(len(filenames))]
    files = sorted(files, key=lambda x: int(x.split("_")[2]))
    for filename in files[:20]:
        filepath = os.path.join(directory, filename)
        try:
            img = Image.open(filepath)
            images.append(img)
        except FileNotFoundError:
            print(f"Error: Image file not found: {filepath}")
            return
        except Exception as e:
            print(f"Error opening image {filename}: {e}")
            return

    if images:
        output_file = os.path.join(directory, "animation.gif")
        imageio.mimsave(output_file, images, duration=duration, loop=loop)
        print(f"GIF created successfully at {output_file}")
    else:
        print("No PNG images found to create a GIF.")


def download_images(run, download_dir="images"):
    """
    Downloads PNG images from a WandB run and saves them to a directory.

    Args:
        run (wandb.run): The WandB run object.
        download_dir (str): The directory to save the downloaded images. Defaults to "images".
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    for file in run.files():
        if file.name.endswith(".png"):
            try:
                file.download(
                    root=download_dir, replace=True
                )  # Specify root and replace
                print(f"Downloaded {file.name} to {download_dir}")
            except Exception as e:
                print(f"Error downloading {file.name}: {e}")


# --- Main ---
if __name__ == "__main__":
    RUN_PATH = "hhayden-mit/Denoiser/6vkdj7jg"  # @param {type:"string"}
    NUM_IMAGES_PER_GIF = 80  # @param {type:"integer"} # Unused variable
    DURATION = 1000  # @param {type:"integer"}
    DOWNLOAD_DIR = "images"  # @param {type:"string"}
    OUTPUT_DIR = "images/media/images/train"  # @param {type:"string"}

    api = wandb.Api()
    try:
        run = api.run(RUN_PATH)
        download_images(run, DOWNLOAD_DIR)
        create_gif(OUTPUT_DIR, DURATION)
    except wandb.CommError as e:
        print(f"Error connecting to WandB API: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
