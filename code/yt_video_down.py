from pytube import YouTube
import moviepy.editor as mp
import cv2
import os

# URL of the YouTube video you want to download
video_url = "https://www.youtube.com/watch?v=Ydiz1Hzfx5s"
audio_save_path = "Indian_Village.mp3"
video_save_path = "Indian_Village.mp4"

# Initialize a YouTube object with the video URL
yt = YouTube(video_url)

# Get the highest resolution video stream
video_stream = yt.streams.get_highest_resolution()

# Specify the directory where you want to save the video


# Download the video to the specified directory
video_stream.download(video_save_path)


clip = mp.AudioFileClip(video_save_path)
clip.write_audiofile(audio_save_path)

# Delete the downloaded video file
clip.close()
os.remove(video_save_path)
