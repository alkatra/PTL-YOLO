from pathlib import Path
from ultralytics import YOLO
import torch
import math
import cv2
import numpy as np
from collections import deque

# Function to detect traffic light color using HSV color space
def detect_light_color_hsv(img, x1, x2, y1, y2):
    # Calculate middle y-coordinate
    middle_y = y1 + ((y2 - y1) // 2)
    # Split the region into upper and lower halves
    upper_half = img[y1:middle_y, x1:x2]
    lower_half = img[middle_y:y2, x1:x2]
    
    # Convert to HSV color space
    upper_hsv = cv2.cvtColor(upper_half, cv2.COLOR_BGR2HSV)
    lower_hsv = cv2.cvtColor(lower_half, cv2.COLOR_BGR2HSV)
    
    # Mask for red color in upper half
    upper_red_mask_1 = cv2.inRange(upper_hsv, (0, 50, 50), (10, 255, 255))
    upper_red_mask_2 = cv2.inRange(upper_hsv, (170, 50, 50), (180, 255, 255))
    upper_red_mask = upper_red_mask_1 | upper_red_mask_2
    
    # Count red pixels
    upper_red_count = np.sum(upper_red_mask == 255)
    
    # Mask for green color in lower half
    lower_green_mask = cv2.inRange(lower_hsv, (25, 50, 50), (100, 255, 255))
    
    # Count green pixels
    lower_green_count = np.sum(lower_green_mask == 255)
    
    # Set pixel thresholds
    upper_pixel_threshold = 0.2 * upper_half.shape[0] * upper_half.shape[1]
    lower_pixel_threshold = 0.1 * lower_half.shape[0] * lower_half.shape[1]
    
    upper_color = "red" if upper_red_count > upper_pixel_threshold else "black"
    lower_color = "green" if lower_green_count > lower_pixel_threshold else "black"
    
    # Final color determination
    if upper_color == "red" and lower_color == "black": return "red" 
    if upper_color == "black" and lower_color == "green": return "green"
    if upper_color == "black" and lower_color == "black": return "off"
    return "unknown/blank"

# Function to extract coordinates of the most confident traffic light detection
def extract_traffic_light_coords(results):
    if results.pred[0] is None:
        return None
    # Filter detections to only include traffic lights (class ID 9)
    traffic_light_detections = [x for x in results.pred[0] if int(x[5]) == 9]
    if not traffic_light_detections:
        return None
    # Sort by confidence and select most confident detection
    most_confident_detection = sorted(traffic_light_detections, key=lambda x: x[4], reverse=True)[0]
    x1, y1, x2, y2 = most_confident_detection[:4]
    return x1, y1, x2, y2

def return_color_values(color):
    if color == "red":
        return (0, 0, 255)
    if color == "green":
        return (0, 255, 0)
    if color == "blinking":
        return (255, 0, 0)
    return (0, 0, 0)

def check_last_three_states(color1, color2, color3):
    if last_three_states[-1] == color1 and last_three_states[-2] == color2 and last_three_states[-3] == color3:
        return True
    return False

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
input_video_path = 'video2.mp4'
output_video_path = 'output.mp4'
# Open video file
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Couldn't open video file.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Initialize video writer
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
i = 0
last_four_colors = deque(['red','red','red','red'], maxlen=4)
last_three_states = deque(['red','red','red'], maxlen=3)
current_state = "red"
red_frames = 0

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    coords = extract_traffic_light_coords(results)
    
    # If traffic light detected, identify its color
    if coords:
        x1, y1, x2, y2 = map(int, coords) # Extract coordinates
        color = detect_light_color_hsv(frame, x1, x2, y1, y2) # Detect color
        if color != "unknown/blank": last_four_colors.append(color) 
        red_frames = red_frames + 1 if color == "red" else 0 # Count consecutive red frames
        # Check if change in color was persistent for 4 frames
        if last_three_states[-1] != color and all(c == last_four_colors[0] for c in last_four_colors): 
            last_three_states.append(color)
        if(last_three_states[-1] == "red"):
            current_state = "red"
        if(last_three_states[-1] == "green"):
            current_state = "green"

        # Traffic light patterns to detect blinking
        blinking = check_last_three_states("off", "red", "off")                 # Red blinking
        blinking = blinking or check_last_three_states("red", "off", "red")     # Red blinking
        blinking = blinking or check_last_three_states("off", "green", "red")   # Initial blinking
        blinking = blinking or check_last_three_states("off", "red", "green")   # Initial blinking

        if blinking:
            current_state = "blinking"
        if red_frames > 15: # Post blinking red
            current_state = "red"
        
        values = return_color_values(current_state)

        # Draw rectangle around traffic light
        if color != "unknown/blank":
            cv2.rectangle(frame, (x1, y1), (x2, y2), values, 2)
            text_position = (x2 + 5, y1 + 5)  # You can adjust this as needed
            cv2.putText(frame, current_state, text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, values, 5)
        
        print("Frame: ", i, "Color: ", color, "State: ", current_state)
        
    # Write frame to output video
    out.write(frame)
    i += 1

# Release resources
cap.release()
out.release()

print(f"Processing complete. Video saved to {Path(output_video_path).resolve()}")
