#Import All the Required Libraries

import cv2
import streamlit as st
from pathlib import Path
import sys
from ultralytics import YOLO
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort

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

#Model Configurations
MODEL_DIR = ROOT/'weights'
DETECTION_MODEL = MODEL_DIR/'yolo11n.pt'

#In case of your custom model
#DETECTION_MODEL = MODEL_DIR/'custom_model_weight.pt'

SEGMENTATION_MODEL  = MODEL_DIR/'yolo11n-seg.pt'

POSE_ESTIMATION_MODEL = MODEL_DIR/'yolo11n-pose.pt'

#Page Layout
st.set_page_config(
    page_title = "YOLO11",
    page_icon = "🤖"
)

# #deepsort
# deepsort = DeepSort()
# # Create a dictionary to store the label mapping
# object_labels = {}

#Header
st.header("Object Tracking")

#SideBar
st.sidebar.header("Model Configurations")

#Choose Model: Detection, Segmentation or Pose Estimation
model_type = "Detection"#st.sidebar.radio("Task", ["Detection"])#, "Segmentation", "Pose Estimation"]) keeping it simple

#Select Confidence Value
confidence_value = float(st.sidebar.slider("Select Model Confidence Value", 25, 100, 40))/100

#Selecting Detection, Segmentation, Pose Estimation Model
if model_type == 'Detection':
    model_path = Path(DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(SEGMENTATION_MODEL)
elif model_type ==  'Pose Estimation':
    model_path = Path(POSE_ESTIMATION_MODEL)

#Load the YOLO Model
try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"Unable to load model. Check the sepcified path: {model_path}")
    st.error(e)

#Image / Video Configuration
st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", SOURCES_LIST
)

def load_video(source_video):
    return cv2.VideoCapture(source_video)

def resize_frame(image, new_width=720):
    height, width = image.shape[:2]
    new_height = int((height / width) * new_width)
    return cv2.resize(image, (new_width, new_height))

def run_object_detection(image, model, confidence):
    result = model.predict(image, conf=confidence, verbose=False)
    return result[0].boxes

def prepare_detections(boxes):
    detections = []
    for box in boxes:
        try:
            coordinates = box.xyxy[0].tolist()
            if len(coordinates) != 4:
                continue
            x1, y1, x2, y2 = coordinates
            conf = box.conf[0].item()
            class_id = int(box.cls[0].item())
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_id))
        except:
            continue
    return detections

def track_objects(detections, image, deepsort):
    return deepsort.update_tracks(detections, frame=image)

def draw_tracks(image, tracks, model, highlight_label, object_labels):
    normal_tracks = []
    highlighted_tracks = []

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        class_id = track.get_det_class()
        class_name = model.names[class_id]

        if track_id not in object_labels:
            object_labels[track_id] = f"{class_name}:{track_id}"

        label = object_labels[track_id]

        if label == highlight_label:
            highlighted_tracks.append((track, label))
        else:
            normal_tracks.append((track, label))

    def draw_single_track(track, label, highlight=False):
        x1, y1, x2, y2 = track.to_ltrb()
        color = (0, 0, 255) if highlight else (255, 0, 0)
        thickness = 2 if highlight else 1

        # Bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        # Text background
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        bg_top_left = (int(x1), int(y1) - text_height - 10)
        bg_bottom_right = (int(x1) + text_width + 4, int(y1))
        cv2.rectangle(image, bg_top_left, bg_bottom_right, color, thickness=cv2.FILLED)

        # Text
        cv2.putText(
            image,
            label,
            (int(x1) + 2, int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # Draw non-highlighted tracks
    for track, label in normal_tracks:
        draw_single_track(track, label)

    # Draw highlighted tracks
    for track, label in highlighted_tracks:
        draw_single_track(track, label, highlight=True)

    return image


source_image = None
if source_radio == IMAGE:
    source_image = st.sidebar.file_uploader(
        "Choose an Image....", type = ("jpg", "png", "jpeg", "bmp", "webp")
    )
    col1, col2 = st.columns(2)
    with col1:
        try:
            if source_image is None:
                default_image_path = str(DEFAULT_IMAGE)
                default_image = Image.open(default_image_path)
                st.image(default_image_path, caption = "Default Image", use_container_width=True)
            else:
                uploaded_image  =Image.open(source_image)
                st.image(source_image, caption = "Uploaded Image", use_container_width = True)
        except Exception as e:
            st.error("Error Occured While Opening the Image")
            st.error(e)
    with col2:
        try:
            if source_image is None:
                default_detected_image_path = str(DEFAULT_DETECT_IMAGE)
                default_detected_image = Image.open(default_detected_image_path)
                st.image(default_detected_image_path, caption = "Detected Image", use_container_width = True)
            else:
                if st.sidebar.button("Detect Objects"):
                    result = model.predict(uploaded_image, conf = confidence_value)
                    boxes = result[0].boxes
                    result_plotted = result[0].plot()[:,:,::-1]
                    st.image(result_plotted, caption = "Detected Image", use_container_width = True)

                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.data)
                    except Exception as e:
                        st.error(e)
        except Exception as e:
            st.error("Error Occured While Opening the Image")
            st.error(e)

elif source_radio == VIDEO:
    source_video = st.sidebar.selectbox(
        "Choose a Video...", VIDEOS_DICT.keys()
    )
    with open(VIDEOS_DICT.get(source_video), 'rb') as video_file:
        video_bytes = video_file.read()
        if video_bytes:
            st.video(video_bytes)
        # Video Detection Logic
        # Move this OUTSIDE the loop so it doesn’t re-render every frame

        # Text input stays outside as before
        highlight_label = st.sidebar.text_input("Enter the label to highlight (e.g., Object1)", "", key="highlight_input")

        # Detect button updates a session state variable
        if st.sidebar.button("Detect Video Objects"):
            st.session_state.run_detection = True

        # Only run detection when the session flag is set
        if st.session_state.get("run_detection", False):
            try:
                video_cap = cv2.VideoCapture(str(VIDEOS_DICT.get(source_video)))
                st_frame = st.empty()
                while video_cap.isOpened():
                    success, image = video_cap.read()
                    if not success:
                        break

                    # Optional: Maintain original aspect ratio
                    height, width = image.shape[:2]
                    new_width = 720
                    new_height = int((height / width) * new_width)
                    image_resized = cv2.resize(image, (new_width, new_height))

                    # Run object detection with YOLO
                    result = model.predict(image_resized, conf=confidence_value)
                    boxes = result[0].boxes

                    # Prepare detections for DeepSORT
                    detections = []
                    for box in boxes:
                        try:
                            coordinates = box.xyxy[0].tolist()
                            if len(coordinates) != 4:
                                st.write(f"Invalid box coordinates: {coordinates}")
                                continue
                            x1, y1, x2, y2 = coordinates
                            conf = box.conf[0].item()
                            class_id = int(box.cls[0].item())
                            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_id))
                        except Exception as e:
                            st.write(f"Error processing box: {e}")
                            continue

                    # DeepSORT tracking
                    try:
                        tracks = deepsort.update_tracks(detections, frame=image_resized)
                    except Exception as e:
                        st.write(f"Error in tracking: {e}")
                        continue

                    # Update and draw tracks
                    for track in tracks:
                        if not track.is_confirmed():
                            continue

                        x1, y1, x2, y2 = track.to_ltrb()
                        track_id = track.track_id

                        # Assign label if not already done
                        if track_id not in object_labels:
                            object_labels[track_id] = f"Object{track_id}"

                        label = object_labels[track_id]

                        # Highlighted object override
                        color = (255, 0, 0)  # Default: Blue
                        thickness = 1

                        if highlight_label and label == highlight_label:
                            color = (0, 255, 0)  # Green
                            thickness = 2

                        # Draw bounding box and label
                        cv2.rectangle(image_resized, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                        cv2.putText(
                            image_resized,
                            label,
                            (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,                  # Smaller font scale
                            (0, 255, 0),          # Color: Light green
                            1,                    # Thinner stroke
                            cv2.LINE_AA           # Better anti-aliasing for smooth text
                        )

                    # Show video frame
                    st_frame.image(image_resized, caption="Detected and Tracked Video", channels="BGR", use_container_width=True)

                video_cap.release()  # Always release after loop ends

            except Exception as e:
                st.sidebar.error(f"Error Loading Video: {e}")
