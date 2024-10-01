import matplotlib.pyplot as plt
from PIL import Image
import imageio
from pathlib import Path

# NOTE: This script assumes that the pngs were generated in an earlier step

# Function to combine two images vertically
def combine_images_vertically(bottom_image_path, top_image_path, output_path: Path):
    bottom_image = Image.open(bottom_image_path)
    top_image = Image.open(top_image_path)

    # Get the width and height of the images
    width1, height1 = bottom_image.size
    width2, height2 = top_image.size

    # Create a new image with the combined height and the maximum width
    combined_img = Image.new('RGB', (max(width1, width2), height1 + height2))

    # Paste the images into the combined image
    combined_img.paste(bottom_image, (0, 0))
    combined_img.paste(top_image, (0, height1))

    # Save the combined image
    combined_img.save(output_path.as_posix())


def get_pngs(indir, pattern: str = "*.png"):
    return list(sorted(indir.glob(pattern)))


def stack_images(bottom_pngs, top_pngs):
    # Read and append each image to the list
    for bottom, top in zip(bottom_pngs, top_pngs):
        image_name = bottom.as_posix().split("/")[-1]
        output_path = cwd / "images_stacked" / "images" / image_name
        combine_images_vertically(bottom, top, output_path)


cwd = Path("./")

indir = cwd / "images_numpy" / "images"
numpy_pngs = get_pngs(indir)

indir = cwd / "images_gpus" / "images"
gpus_pngs = get_pngs(indir)

assert len(gpus_pngs) == len(numpy_pngs)

print(f"Number of images to combine: {len(gpus_pngs)}")

create_stacked_images: bool = True 
create_stacked_gif: bool = True 

if create_stacked_images:
    stack_images(gpus_pngs, numpy_pngs)

if create_stacked_gif:

    # Create a list to store the images
    indir = cwd / "images_stacked" / "images"
    stacked_images = get_pngs(indir)

    # Make gif
    images = []
    for image in stacked_images:
        images.append(imageio.imread(image))

    # Save the images as an animated gif
    output_path = cwd / "images_stacked" / "images" / "stacked.gif" 
    imageio.mimsave(output_path, images)
