import os
import shutil
import torch
import torch.nn as nn
from skimage.color import lab2rgb
import re
import numpy as np
import moviepy.editor as mp
import cv2
from pytube import YouTube

def walk_through_dir(dir_path):
    """Walks through dir_path returning its contents."""
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'."
        )


def get_file_path(dir_path):
    file_paths = []
    for root, _, files in os.walk(dir_path):
        for filename in files:
            file_paths.append(os.path.join(root, filename))
    return file_paths


def split_data(
    train_file_path, test_file_path, train_final_destination, test_final_destination
):

    for i in train_file_path:

        shutil.copy(i, train_final_destination)
    for i in test_file_path:

        shutil.copy(i, test_final_destination)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L + 1.0) * 50.0
    ab = ab * 110.0
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def get_frame_number(file_path):
    # Extract the filename from the full path
    filename = file_path.split("/")[-1]
    # Use regular expression to find the number
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"No number found in file path: {file_path}")



def download_yt(video_url, video_save_path, audio_save_path):
    yt = YouTube(video_url)
    video_stream = yt.streams.get_highest_resolution()
    video_stream.download(video_save_path)
    clip = mp.AudioFileClip(video_save_path)
    clip.write_audiofile(audio_save_path)
