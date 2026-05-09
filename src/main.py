import cv2
import os
from detection import load_model, load_video, detect_players
from shot_classification import classify_shot, get_player_label
from analytics import save_results

# ---- Setup ----

# Load the pretrained YOLOv8 model
model = load_model("models/yolov8n.pt")

# Load the input video and get its properties
cap, fps, width, height = load_video("data/sample_video.mp4")

print(f"Video loaded successfully!")
print(f"FPS: {fps}, Resolution: {width}x{height}")

# Create output folder if it doesn't exist
os.makedirs("output", exist_ok=True)

# Setup output video writer with same fps and resolution as input
out = cv2.VideoWriter("output/annotated_video.mp4",
                       cv2.VideoWriter_fourcc(*"mp4v"),
                       fps, (width, height))

# List to store all detection results for saving to CSV/JSON later
results_list = []

# Counter to track current frame number
frame_number = 0

# Color mapping for each shot type (BGR format)
# Green = Forehand, Blue = Backhand, Red = Smash, Gray = Neutral
shot_colors = {
    "Forehand": (0, 255, 0),
    "Backhand": (255, 0, 0),
    "Smash": (0, 0, 255),
    "Neutral": (200, 200, 200)
}

print("Processing video... this will take a few minutes, please wait.")

# ---- Main Processing Loop ----

while True:
    # Read the next frame from the video
    ret, frame = cap.read()
    
    # If no more frames, we've reached the end of the video
    if not ret:
        break

    # Only run detection every 5th frame to speed up processing
    if frame_number % 5 == 0:
        
        # Detect all players in this frame
        players = detect_players(model, frame)

        for (x1, y1, x2, y2, confidence) in players:
            
            # Find the center of the player's bounding box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Identify which player this is based on court position
            player = get_player_label(center_x, center_y, width, height)
            
            # Classify shot based on player movement since last frame
            shot_type = classify_shot(player, center_x, center_y, width, height)
            
            # Get the color for this shot type
            color = shot_colors[shot_type]

            # Draw colored bounding box around the player
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label above the box showing player and shot type
            cv2.putText(frame, f"{player} | {shot_type}",
                       (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, color, 2)

            # Only save actual shots, not neutral positions
            if shot_type != "Neutral":
                timestamp = round(frame_number / fps, 2)
                results_list.append({
                    "frame": frame_number,
                    "timestamp": timestamp,
                    "player": player,
                    "shot_type": shot_type,
                    "confidence": round(confidence, 2)
                })

    # Write this frame to output video
    out.write(frame)
    
    # Move to next frame
    frame_number += 1

# ---- Cleanup ----
cap.release()
out.release()

# Save results and print analytics
save_results(results_list)
print(f"\nDone! Processed {frame_number} frames.")