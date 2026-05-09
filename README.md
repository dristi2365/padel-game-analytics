# Padel Game Analytics — Shot Classification System

A computer vision system that analyzes padel match footage to detect players and classify shot types using YOLOv8 and OpenCV.

Built as part of the Layman AI — AI/ML Internship Technical Assessment.

---

## What it does

- Detects and tracks players in a padel match video
- Classifies each player's shots as Forehand, Backhand, or Smash
- Outputs an annotated video with colored bounding boxes and labels
- Saves all detections to CSV and JSON formats
- Generates shot analytics including breakdown per player

---

## How it works

### Player Detection
YOLOv8 (yolov8n pretrained model) is used to detect people in each frame. The model runs on every 5th frame to balance speed and accuracy. Only detections with confidence above 0.35 are kept.

### Player Identification
The court is divided into 4 quadrants to assign player labels:
Player 1 (top-left)Player 2 (top-right)Player 3 (bot-left)Player 4 (bot-right)

### Shot Classification
A motion-based rule approach is used. For each player, we track their center position between frames and calculate movement:

- Moving upward fast (dy < -15) → **Smash**
- Moving right (dx > 10) → **Forehand**
- Moving left (dx < -10) → **Backhand**
- Barely moving (total movement < 8) → **Neutral** (not saved)

This is more accurate than purely position-based rules because it uses actual player movement rather than fixed court zones.

### Output Colors
- 🟢 Green = Forehand
- 🔵 Blue = Backhand  
- 🔴 Red = Smash
- ⚪ Gray = Neutral

---

## Project Structure
padel_project/
data/
sample_video.mp4        # Input padel match video
models/
yolov8n.pt              # Pretrained YOLOv8 model
output/
annotated_video.mp4     # Output video with annotations
shots.csv               # Detection results in CSV format
shots.json              # Detection results in JSON format
src/
main.py                 # Main script - runs the full pipeline
detection.py            # Player detection using YOLOv8
shot_classification.py      # Motion based shot classification
analytics.py            # Saves results and prints summary
requirements.txt
README.md

---

## Setup and Installation

**Requirements:** Python 3.11+

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/dristi2365/padel-game-analytics.git
cd padel-game-analytics
```

**Step 2 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 3 — Add your video:**
Place your padel match video in the `data/` folder and name it `sample_video.mp4`

**Step 4 — Run:**
```bash
python src/main.py
```

The YOLOv8 model will download automatically on first run.

---

## Results

Sample output from the test video (325 seconds, 8125 frames):

| Shot Type | Count |
|-----------|-------|
| Smash     | 1113  |
| Backhand  | 1011  |
| Forehand  |  612  |

| Player   | Backhand | Forehand | Smash |
|----------|----------|----------|-------|
| Player 1 | 594      | 141      | 600   |
| Player 2 | 202      | 314      | 378   |
| Player 3 | 155      | 105      | 87    |
| Player 4 | 60       | 52       | 48    |

---

## Limitations

- **Far-side player detection** is inconsistent due to camera distance and angle — players appear smaller and are harder for YOLO to detect reliably
- **Near-side players** are sometimes partially hidden behind the court fence, causing missed detections
- **Shot classification** is rule-based and approximate — it uses movement direction rather than actual body/arm pose, which means some misclassifications occur
- **Player identification** uses fixed court quadrants — if players cross sides, labels may swap

---

## Challenges Faced

- **Camera angle** — the overhead fixed camera made it difficult to detect players 
on the far side of the court as they appear much smaller in the frame

- **Fence obstruction** — near-side players are partially hidden behind the court 
fence at the bottom of the frame, causing missed or incomplete detections

- **Shot classification without pose data** — without being able to see arm and 
body orientation clearly from above, classifying shots accurately was the biggest 
challenge. My first approach used fixed court positions which produced unrealistic 
results (players only getting one shot type). I switched to a motion-based approach 
tracking player movement between frames which gave much more realistic results

- **No prior knowledge of padel** — I had never analyzed padel footage before this 
assignment. Understanding what constitutes each shot type required research and 
experimentation

- **Processing speed** — running YOLO on every frame of a 5 minute 1080p video 
was too slow, so I optimized by processing every 5th frame which gave a good 
balance between speed and accuracy

---

## What I would improve with more time

- **Pose estimation using MediaPipe** — detect actual arm and body orientation to classify shots more accurately instead of relying on movement direction
- **Use a larger YOLO model** (yolov8m or yolov8l) for better detection accuracy, especially for far-side players
- **Multiple camera angles** — the fixed overhead angle misses a lot. Multiple cameras would dramatically improve detection
- **Study padel more deeply** — understanding the sport's biomechanics would help design better classification rules
- **Train a custom shot classifier** — with enough labeled padel footage, a dedicated ML model could learn shot patterns automatically

---

## Tech Stack

- Python 3.11
- YOLOv8 (Ultralytics)
- OpenCV
- Pandas
- NumPy

---

## Author

Dristi Shakya
shakyadristi2@gmail.com
[LinkedIn](https://www.linkedin.com/in/dristi-shakya-439908374/)