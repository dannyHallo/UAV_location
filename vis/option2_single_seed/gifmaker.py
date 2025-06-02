import os
from PIL import Image


def create_gif_from_pngs(output_gif_name="output.gif", duration=50, step=10):
    """
    Create a GIF from PNG files in the current folder, using every `step`th image.

    Parameters:
    - output_gif_name: Name of the output GIF file.
    - duration: Duration of each frame in milliseconds.
    - step: Select every `step`th image from the sorted list.
    """
    # Get the current directory
    current_dir = os.getcwd()

    # Get all PNG files in the current directory
    png_files = [f for f in os.listdir(current_dir) if f.lower().endswith(".png")]

    # Sort PNG files by modification time (oldest first)
    png_files.sort(key=lambda f: os.path.getmtime(os.path.join(current_dir, f)))

    if not png_files:
        print("No PNG files found in the current directory.")
        return

    # Filter the PNG files to use every `step`th image
    selected_files = png_files[::step]

    # Open images and store them in a list
    images = []
    for file in selected_files:
        img = Image.open(os.path.join(current_dir, file))
        images.append(img)

    if not images:
        print("No images selected. Check your step value.")
        return

    # Save the images as a GIF
    images[0].save(
        output_gif_name,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=duration,
        loop=0,
    )
    print(f"GIF saved as {output_gif_name}")


# Call the function with a step value of 10
create_gif_from_pngs(step=5)
