import cv2
from ultralytics import YOLO
import pandas as pd
import os

# Load YOLO model
model = YOLO("models/yolov8n.pt")

# Path to the video
video_path = "data/sample_video.mp4"

# Open the video
cap = cv2.VideoCapture(video_path)

# Read the basic informations about the video
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video loaded successfully!")
print(f"FPS: {fps}")
print(f"Resolution: {width}*{height}")

# Prepare to save output video
os.makedirs("output", exist_ok=True)
out = cv2.VideoWriter("output/annotated_video.AVI", cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

#Store results
results_list = []
frame_number = 0

print("Processing video... this will take a few minutes, please wait.")

#Loop through every frame
while True:
    ret, frame = cap.read()

    #If no more frames, stop
    if not ret:
        break

    #Only process every 5th frame to save time
    if frame_number % 5 == 0:

        #Run YOLO on this frame
        results = model(frame, verbose=False)

        # Go through each detection
        for result in results:
            for box in result.boxes:

                # Class 0 means "person" in YOLO
                if int(box.cls) == 0:

                    # Get the box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Get confidence score
                    confidence = float(box.conf[0])

                    # Only keep detections we are confident about
                    if confidence > 0.5:

                        # Draw a green box around the person
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        #Add a label above the box
                        cv2.putText(frame, f"Player {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        #Save this detection to our list
                        timestamp = round(frame_number / fps, 2)
                        results_list.append({
                            "frame": frame_number,
                            "timestamp": timestamp,
                            "x1": x1, "y1": y1,
                            "x2": x2, "y2": y2,
                            "confidence": round(confidence, 2)
                        })

    # Write frame to output video
    out.write(frame)
    frame_number += 1

# Clean up
cap.release() 
out.release()  

#Save results to CSV
df = pd.DataFrame(results_list)
df.to_csv("output/shots.csv", index=False)

print(f"Done! Processed {frame_number} frames.")
print(f"Detected {len(results_list)} players across all frmaes.")
print(f"Results saved to output/shots.csv")
print(f"Annotated video saved to output/annotated_video.mp4")