import cv2
import numpy as np
import sys
import supervision as sv
from ultralytics import YOLO
from fastapi import FastAPI
from threading import Thread
import uvicorn

# Initialize the YOLO model, ByteTrack tracker, and box annotator
model = YOLO('vehicle_counter/yolov8n.pt')
#model=YOLO('bbest.pt')

byte_tracker = sv.ByteTrack()
annotator = sv.BoxAnnotator()

# Step 1: Open a file to write the results
result_file = open("detection_tracking_results.txt", "w")
result_file1 = open("detection_tracking_results1.txt", "w")

count = 0

side_settings = {
    'side1': {
        'y_up': 200, 'y_down': 300,
        'total_count_up': 0, 'total_count_down': 0,
        'current_vehicles': 0,  # Initialize the number of current vehicles

        'up': set(), 'down': set(),
        'counts': {cid: {'up': 0, 'down': 0} for cid in [2, 3, 5, 7]}
    },
    'side2': {
        'y_up': 150, 'y_down': 250,
        'total_count_up': 0, 'total_count_down': 0,
        'current_vehicles': 0,  # Initialize the number of current vehicles

        'up': set(), 'down': set(),
        'counts': {cid: {'up': 0, 'down': 0} for cid in [2, 3, 5, 7]}
    }
}

id_to_last_position = {}

def count_vehicles(side, id, class_id, cx, cy, last_pos):
    settings = side_settings[side]
    y_up = settings['y_up']
    y_down = settings['y_down']

    current_up_pos = 'above' if cy < y_up else 'below'
    current_down_pos = 'above' if cy < y_down else 'below'
    
    # Increment current vehicles when crossing the first line, regardless of direction
    if (last_pos[0] == 'below' and current_up_pos == 'above') or (last_pos[0] == 'above' and current_up_pos == 'below'):
        settings['current_vehicles'] += 1
        settings['up'].add(id)  # Track IDs that have crossed this line

    # Decrement current vehicles when crossing the second line, regardless of direction
    if (last_pos[1] == 'above' and current_down_pos == 'below') or (last_pos[1] == 'below' and current_down_pos == 'above'):
        settings['current_vehicles'] -= 1
        settings['down'].add(id)  # Track IDs that have crossed this line

    if last_pos[0] == 'below' and current_up_pos == 'above' and id not in settings['up']:
        settings['total_count_up'] += 1
        settings['up'].add(id)
    if last_pos[0] == 'above' and current_up_pos == 'below' and id not in settings['down']:
        settings['total_count_down'] += 1
        settings['down'].add(id)
    if last_pos[1] == 'above' and current_down_pos == 'below' and id not in settings['down']:
        settings['total_count_down'] += 1
        settings['down'].add(id)
    if last_pos[1] == 'below' and current_down_pos == 'above' and id not in settings['up']:
        settings['total_count_up'] += 1
        settings['up'].add(id)

    # Update specific class counts
    if class_id in [2, 3, 5, 7]:
        if last_pos[0] == 'below' and current_up_pos == 'above':
            settings['counts'][class_id]['up'] += 1
        if last_pos[0] == 'above' and current_down_pos == 'below':
            settings['counts'][class_id]['down'] += 1            
        if last_pos[1] == 'above' and current_down_pos == 'below':
            settings['counts'][class_id]['down'] += 1
        if last_pos[1] == 'below' and current_up_pos == 'above':
            settings['counts'][class_id]['up'] += 1

    id_to_last_position[id] = (current_up_pos, current_down_pos)

def draw_side_info(frame, side, line_coords, text_coords):
    color = (160, 168, 50)  # Olive green color in BGR
    transparency = 0.5  # 50% transparency
    thickness = 90

    overlay = np.zeros_like(frame)
    mask = np.zeros_like(frame, dtype=np.uint8)

    for start, end in line_coords:
        cv2.line(overlay, start, end, color, thickness)
        cv2.line(mask, start, end, (255, 255, 255), thickness)

    mask = mask.astype(bool)
    frame[mask] = cv2.addWeighted(frame, 1 - transparency, overlay, transparency, 0)[mask]

    for text, position in text_coords:
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (position[0], position[1] - text_size[1]), 
                      (position[0] + text_size[0], position[1]), (255, 255, 255), -1)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    settings = side_settings[side]
    x_position = 180 if side == 'side1' else 370
    y_offset = 410 if side == 'side1' else 360

    current_vehicles_text = f"Current Vehicles: {settings['current_vehicles']}"
    cv2.putText(frame, current_vehicles_text, (x_position, y_offset - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    for i, cid in enumerate([2, 3, 5, 7]):
        up_text = f"Class {cid} Up: {settings['counts'][cid]['up']}"
        down_text = f"Class {cid} Down: {settings['counts'][cid]['down']}"
        cv2.putText(frame, up_text, (x_position, y_offset + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, down_text, (x_position, y_offset + 10 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def process_frame(frame: np.ndarray, index: int, file, file1) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = byte_tracker.update_with_detections(detections)

    for xyxy, confidence, class_id, tracker_id in zip(detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id):
        if class_id in [2, 3, 5, 7]:  # Filtering for specific classes
            file.write(f"Frame {index}: ID #{tracker_id}, Class: {model.model.names[class_id]}, Confidence: {confidence:.2f}, Box: {list(xyxy)}\n")
            id = tracker_id
            cx = int((xyxy[0] + xyxy[2]) / 2)  # Calculate center x of the box
            cy = int((xyxy[1] + xyxy[3]) / 2)  # Calculate center y of the box
            cv2.circle(frame, (int(cx), int(cy)), 3, (0, 100, 255), -1)                  
            last_pos = id_to_last_position.get(id, (None, None))
            side = 'side1' if cx < frame.shape[1] // 2 else 'side2'
            count_vehicles(side, id, class_id, cx, cy, last_pos)

    file1.write(f"Frame {index} Side1 Counts: Total Up: {side_settings['side1']['total_count_up']}, Total Down: {side_settings['side1']['total_count_down']}\n")
    file1.write(f"Frame {index} Side2 Counts: Total Up: {side_settings['side2']['total_count_up']}, Total Down: {side_settings['side2']['total_count_down']}\n")
    for side in ['side1', 'side2']:
        settings = side_settings[side]
        file1.write(f"Frame {index} {side} Counts: " +
            ", ".join([f"Class {cid} Up: {settings['counts'][cid]['up']}, Down: {settings['counts'][cid]['down']}" for cid in [2, 3, 5, 7]]) + "\n")

    # Drawing lines and info for side1
    draw_side_info(frame, 'side1',
                   [((0, 200), (300, 200)), ((0, 300), (300, 300))],
                   [(f"Up: {side_settings['side1']['total_count_up']}", (10, 420)),
                    (f"Down: {side_settings['side1']['total_count_down']}", (10, 440))])

    # Drawing lines and info for side2
    draw_side_info(frame, 'side2',
                   [((400, 150), (700, 150)), ((400, 250), (700, 250))],
                   [(f"Up: {side_settings['side2']['total_count_up']}", (410, 370)),
                    (f"Down: {side_settings['side2']['total_count_down']}", (410, 390))])

    return annotator.annotate(scene=frame, detections=detections)

app = FastAPI()

@app.get("/count")
def get_count():
    result = {
        'side1': {
            'total_up': side_settings['side1']['total_count_up'],
            'total_down': side_settings['side1']['total_count_down'],
            'current_vehicles': side_settings['side1']['current_vehicles'],
            'counts': side_settings['side1']['counts']
        },
        'side2': {
            'total_up': side_settings['side2']['total_count_up'],
            'total_down': side_settings['side2']['total_count_down'],
            'current_vehicles': side_settings['side2']['current_vehicles'],
            'counts': side_settings['side2']['counts']
        }
    }
    return result

def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    video_capture = cv2.VideoCapture(sys.argv[1])

    if not video_capture.isOpened():
        print("Error: Could not open video.")
        sys.exit()

    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(
        sys.argv[2],
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (frame_width, frame_height),
    )

    fastapi_thread = Thread(target=run_fastapi)
    fastapi_thread.start()

    index = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = process_frame(frame, index, result_file, result_file1)
        video_writer.write(frame)
        index += 1

    video_capture.release()
    video_writer.release()
    result_file.close()
    result_file1.close()
