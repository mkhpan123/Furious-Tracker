#Import All the Required Libraries
import cv2
import streamlit as st
from pathlib import Path
import sys
from ultralytics import YOLO
from PIL import Image

#Get the absolute path of the current file
FILE = Path(__file__).resolve()

#Get the parent directory of the current file
ROOT = FILE.parent

#Add the root path to the sys.path list
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

#Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

#Sources
IMAGE = 'Image'
VIDEO = 'Video'

SOURCES_LIST = [IMAGE, VIDEO]

#Image Config
IMAGES_DIR = ROOT/'images'
DEFAULT_IMAGE = IMAGES_DIR/'image1.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR/'detectedimage1.jpg'

#Videos Config
VIDEO_DIR = ROOT/'videos'
VIDEOS_DICT = {
    'video 1': VIDEO_DIR/'video1.mp4',
    'video 2': VIDEO_DIR/'video2.mp4'
}

