import av
import cv2
import numpy as np
import streamlit as st
import requests
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.title("Live Video Streaming")

BACKEND_URL = "http://127.0.0.1:5000/process_frame"  # Backend endpoint

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        processed_img = process_video_frame(img)  # Send to backend and receive processed frame
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

def process_video_frame(frame):
    """
    Function to send video frame to backend and receive processed frame.
    """
    _, img_encoded = cv2.imencode('.jpg', frame)  # Encode frame as JPEG
    response = requests.post(BACKEND_URL, files={"file": img_encoded.tobytes()})
    
    if response.status_code == 200:
        nparr = np.frombuffer(response.content, np.uint8)
        processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return processed_frame
    else:
        return frame  # Return original frame if backend fails

webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
