# Importing necessary libraries
from ultralytics import YOLO
from pathlib import Path
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
    upper_pixel_threshold = 0.1 * upper_half.shape[0] * upper_half.shape[1]
    lower_pixel_threshold = 0.1 * lower_half.shape[0] * lower_half.shape[1]
    
    # print(upper_red_count / (upper_half.shape[0] * upper_half.shape[1]))
    upper_color = "red" if upper_red_count > upper_pixel_threshold else "black"
    lower_color = "green" if lower_green_count > lower_pixel_threshold else "black"
    # Final color determination
    if upper_color == "red" and lower_color == "black": return "red" 
    if upper_color == "black" and lower_color == "green": return "green"
    if upper_color == "black" and lower_color == "black": return "off"
    return "off"

# Function to extract the coordinates of the most confident traffic light detection from model results
def extract_traffic_light_coords(results):
    # Find the indices of traffic light detections (class id for traffic light is 9 in YOLOv8)
    traffic_lights = (results[0].boxes.cls == 0.).nonzero(as_tuple=True)[0]
    
    # If no traffic lights were detected, print a message and exit the function
    if len(traffic_lights) == 0:
        # print("No traffic light found.")
        return None

    # Extract the confidence scores of these detections
    confidence_values = results[0].boxes.conf[traffic_lights]
    
    # Get the index of the detection with the highest confidence
    most_confident_light, idx_traffic_light = torch.max(confidence_values, 0)
    
    # Extract and round off the confidence value
    confidence = math.floor(most_confident_light.item() * 100) / 100
    
    # Get the bounding box of the most confident detection
    box = results[idx_traffic_light].boxes.xyxy[idx_traffic_light]
    x1, y1, x2, y2 = map(int, box)
    return x1, y1, x2, y2, confidence *100

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

# Custom trained YOLO Model
model = YOLO("best.pt") 
input_video_path = 'video3.mp4'
output_video_path = 'output3.mp4'
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Couldn't open video file.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Initialize video writer
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
i = 0
last_four_colors = deque(['red','red','red','red'], maxlen=4)
last_three_states = deque(['red','red','red'], maxlen=3)
traffic_state = "red"
red_frames = 0
currently_blinking = False
# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(source=frame, verbose=False)
    # print(results[0].boxes.cls)
    coords = extract_traffic_light_coords(results)
    if coords:
        x1, y1, x2, y2, confidence = map(int, coords) # Extract coordinates
        color = detect_light_color_hsv(frame, x1, x2, y1, y2) # Detect color
    else: # Traffic light assumed to be off if no detections present
        color="off"
    last_four_colors.append(color) 
    red_frames = red_frames + 1 if color == "red" else 0 # Count consecutive red frames
    # Check if change in color was persistent for 4 frames
    if last_three_states[-1] != color and all(c == last_four_colors[0] for c in last_four_colors): 
        last_three_states.append(color)
    if(last_three_states[-1] == "red"):
        traffic_state = "red"
    if(last_three_states[-1] == "green"):
        traffic_state = "green"
    blinking = False
    # Traffic light patterns to detect blinking
    if last_three_states[-2] == "green" and (last_three_states[-1] == "red"):
        currently_blinking = True # Only activate blinking if the light was green before turning red
        traffic_state = "blinking"
        blinking = True
    if currently_blinking:
        blinking = check_last_three_states("off", "red", "off")                 
        blinking = blinking or check_last_three_states("red", "off", "red")     
        blinking = blinking or check_last_three_states("off", "green", "red")   
        blinking = blinking or check_last_three_states("off", "red", "green")   
        blinking = blinking or check_last_three_states("off", "green", "off")  

    if blinking:
        traffic_state = "blinking"
    if red_frames > 20: # Post blinking red
        traffic_state = "red"
        currently_blinking = False
    
    values = return_color_values(traffic_state)

    # Draw rectangle around traffic light
    cv2.rectangle(frame, (x1, y1), (x2, y2), values, 2)
    text_position = (x2 + 5, y1 + 5)  # You can adjust this as needed
    cv2.putText(frame, traffic_state, text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, values, 5)
    
    message = "Frame: " + str(i) + "/" + str(framecount) + " Color: " + color + " State: " + traffic_state + " Confidence: " + str(confidence) + "%"
    print(f'{message}\r', end='', flush=True)
        
    # Write frame to output video
    out.write(frame)
    i += 1

# Release resources
cap.release()
out.release()
print(f"Processing complete. Video saved to {Path(output_video_path).resolve()}")
