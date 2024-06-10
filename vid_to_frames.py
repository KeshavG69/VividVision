import os
import utils
from generator import Generator
import torch
from dataloaders import Colorization, datalaoder
import numpy as np
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"

output_folder = "/content/drive/MyDrive/Colorize/Frames"
video_path = "/content/drive/MyDrive/Colorize/Indian_Village.mp4"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

"""Run this in shell to  convert video to frames"""

"""!ffmpeg -i /content/drive/MyDrive/Colorize/Indian_Village.mp4 -vf fps=30 /content/drive/MyDrive/Colorize/Frames/frame-%04d.png"""

frames = utils.get_file_path(output_folder)

frames = sorted(frames, key=utils.get_frame_number)

frames_dataset = Colorization(frames, size=256, split="test")

frames_dataloader = datalaoder(dataset=frames_dataset, BATCH_SIZE=32, shuffle=False)

gen = Generator(in_channels=1).to(device)

gen.load_state_dict(
    torch.load(
        "/content/drive/MyDrive/Colorize/gen_model_30_face_25_landscape.pth",
        map_location=torch.device(device),
    )
)

gen.eval()

fake_images = []  # List to store generated fake RGB images
real_images = []  # List to store real RGB images
for idx, batch in enumerate(
    frames_dataloader
):  # Assuming your data loader yields batches of L and AB channels
    print(idx)
    L = batch["L"].to(device)
    ab = batch["ab"].to(device)
    fake_color = gen(L)  # Generate fake color predictions
    real_color = ab
    fake_image = utils.lab_to_rgb(L, fake_color)  # Convert fake color to RGB
    real_image = utils.lab_to_rgb(L, real_color)
    fake_images.append(fake_image)  # Append fake RGB image to list
    real_images.append(real_image)

fake_images = np.vstack(fake_images)
real_images = np.vstack(real_images)

output_dir = "/content/drive/MyDrive/Colorize/Coloured_Frames"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

fake_images_uint8 = (fake_images * 255).astype(np.uint8)

for i, img in enumerate(fake_images_uint8):
    # Create the filename with leading zeros
    filename = f"{i+1:04}.jpg"
    filepath = os.path.join(output_dir, filename)

    # Convert the NumPy array to a PIL Image and save it
    image = Image.fromarray(img)
    image.save(filepath)

    print(f"Saved {filepath}")

"""Run this in shell to convert frames to video with audio"""
"""!ffmpeg -framerate 30 -i /content/drive/MyDrive/Colorize/Coloured_Frames/%04d.jpg -i /content/drive/MyDrive/Colorize/Indian_Village.mp3 -c:v libx264 -c:a aac -r 30 -pix_fmt yuv420p -shortest /content/drive/MyDrive/Colorize/output_video.mp4"""
