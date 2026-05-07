import cv2
from ultralytics import YOLO
import pandas as pd
import os
import json

# Load the YOLO model 
model = YOLO("models/yolov8n.pt")

#Path to the video
video_path = "data/sample_video.mp4"

#Open the video
cap = cv2.VideoCapture(video_path)

# Read basic information about the video
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video loaded successfully!")
print(f"FPS: {fps}, Resolution: {width}*{height}")

# Pepare to save output video
os.makedirs("output", exist_ok=True)
out = cv2.VideoWriter("output/annotated_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

#Store results
results_list = []
frame_number = 0

# New: Shot classification function
def classify_shot(x1, y1, x2, y2, width, height):
    #Find the center of the player box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    #If player is in the top 30% of the frame (back of court) -> smash
    if center_y < height * 0.3:
        return "Smash"
    
    #If player is in bottom half of the frame
    # and on the right side -> forehand
    elif center_x > width * 0.5:
        return "Forehand"
    
    #Otherwise-> backhand
    else:
        return "Backhand"
    
# Player labeling function
def get_player_label(center_x, center_y, width, height):
    # Top half of court = Team A, Bottom half = Team B
    if center_y < height * 0.5:
        if center_x < width * 0.5:
            return "Player1"
        else:
            return "Player2"
    else:
        if center_x < width * 0.5:
            return "Player3"
        else:
            return "Player4"
            
print("Processing video..... thsi will take a few minutes, please wait.")

#Colors for each shot type
shot_colors = {
    "Forehand": (0, 255, 0), #Green
    "Backhand": (255, 0, 0), #Blue
    "Smash": (0, 0, 255) #Red
}
    
# Loop though every frame
while True:
    ret, frame = cap.read()

    # If no more frames, stop
    if not ret:
        break

    # Only process every 5th frame to save time
    if frame_number % 5 == 0:

        # Run YOLO on this frame
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

                    # Only keep detections we're confident about
                    if confidence > 0.5:

                        #classify the shot
                        shot_type = classify_shot(x1, y1, x2, y2, width, height)

                        #identify the player
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        player = get_player_label(center_x, center_y, width, height)

                        # use different color per shot type
                        color = shot_colors[shot_type]

                        #Draw box with short colors
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        #show player name and shot type
                        label = f"{player} | {shot_type}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        #Save to the results
                        timestamp = round(frame_number / fps, 2)
                        results_list.append({
                            "frame": frame_number,
                            "timestamp": timestamp,
                            "player": player,
                            "shot_type": shot_type,
                            "confidence": round(confidence, 2)
                        })
    
    out.write(frame)
    frame_number += 1

#Clean up
cap.release()
out.release()

# Save to csv
df = pd.DataFrame(results_list)
df.to_csv("output/shots.csv", index=False)

#save to json
with open("output/shots.json", "w") as f:
    json.dump(results_list, f, indent=4)

# print shot analytics summary
print(f"\nDone! Processed {frame_number} frames.")
print(f"\n--- Shot Analytics ---")
print(df["shot_type"].value_counts().to_string())
print(f"\n--- Shots per Player ---")
print(df["player"].value_counts().to_string())
print("\n--- Shot Breakdown per Player ---")
breakdown = df.groupby(["player", "shot_type"]).size().unstack(fill_value=0)
print(breakdown.to_string())
print(f"\nFiles saved to output/ folder")