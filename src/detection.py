import cv2
from ultralytics import YOLO

def load_model(model_path):
    # Load the pretrained YOLOv8 from the given path
    return YOLO(model_path)

def load_video(video_path):
    #Open the video file for reading
    cap = cv2.VideoCapture(video_path)

    #Extract basic video properties needed for processing and output
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return cap, fps, width, height

def detect_players(model, frame):
    #Run YOLO detection on the current frame
    #verbose=False suppresses per-frame console output
    results = model(frame, verbose=False)

    players = []

    for result in results:
        for box in result.boxes:
            # Class 0 in YOLO's COCO dataset corresponds to 'person'
            # We only care about people, not other detected objects
            if int(box.cls) == 0:
                #Get bounding box coordinates (top-left and bottom-right corners)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get the confidence score (how sure YOLO is if this is a person)
                confidence = float(box.conf[0])

                # Lowered threshold to catch partially visible players
                #near the fence at the bottom of the frame
                if confidence > 0.35:
                    players.append((x1, y1, x2, y2, confidence))
    return players