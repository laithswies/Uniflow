import supervision as sv
import numpy as np
from ultralytics import YOLO
import cv2

model = YOLO('/home/yoda/uniflow/YOLOBTRACK/yolov8n.pt')

#Initializes an instance of the ByteTrack algorithm
byte_tracker = sv.ByteTrack()

#object that can draw bounding boxes and labels on images. This is used to annotate frames with detection and tracking information
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

def count_vehicles(side, id, class_id,cx, cy, last_pos):
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
    # Draw lines on the frame
  
    color = (160, 168, 50)  # Olive green color in BGR
    transparency = 0.5  # 50% transparency
    thickness = 90

  # Create an overlay and a mask of the same size as the frame, initialized to zero
    overlay = np.zeros_like(frame)
    mask = np.zeros_like(frame, dtype=np.uint8)

  # Draw the lines on the overlay and mask
    for start, end in line_coords:
          cv2.line(overlay, start, end, color, thickness)
          cv2.line(mask, start, end, (255, 255, 255), thickness)

  # Convert the mask to a boolean mask and apply it to blend only the lines
    mask = mask.astype(bool)
    frame[mask] = cv2.addWeighted(frame, 1 - transparency, overlay, transparency, 0)[mask]
 
  
  
    # Update text on the frame
    for text, position in text_coords:
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (position[0], position[1] - text_size[1]), 
                      (position[0] + text_size[0], position[1]), (255, 255, 255), -1)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Additional specific class counts display
    settings = side_settings[side]
    x_position = 180 if side == 'side1' else 370
    y_offset = 410 if side == 'side1' else 360  # Starting y position for specific counts
        
        # Display the current vehicle count
    current_vehicles_text = f"Current Vehicles: {settings['current_vehicles']}"
    cv2.putText(frame, current_vehicles_text, (x_position, y_offset - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    for i, cid in enumerate([2, 3, 5, 7]):
        up_text = f"Class {cid} Up: {settings['counts'][cid]['up']}"
        down_text = f"Class {cid} Down: {settings['counts'][cid]['down']}"
        cv2.putText(frame, up_text, (x_position, y_offset + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, down_text, (x_position, y_offset + 10 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        

#function processes individual frames of video. It takes a frame (as a NumPy array) and an index indicating the frame number.
def callback(frame: np.ndarray, index: int,file,file1) -> np.ndarray:
    #Passes the current frame to the YOLO model to perform object detection. The model returns detection results, typically including bounding boxes, confidence scores, and class IDs.
    results = model(frame)[0]
    #Converts the raw detection results from the YOLO model into a format that can be used by the ByteTrack tracker
    #parsing the results to extract relevant data such as coordinates, confidence levels, and class identifiers.
    detections = sv.Detections.from_ultralytics(results)
    #Updates the ByteTrack tracker with the new detections from the current frame. The tracker adjusts the object tracks based on the new information, helping to maintain continuity of object identities across frames.
    detections = byte_tracker.update_with_detections(detections)
    
 
        # Write detection and tracking results to file
    for xyxy, confidence, class_id, tracker_id in zip(detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id):
        if class_id in [2, 3, 5, 7]:  # Filtering for specific classes

        
            file.write(f"Frame {index}: ID #{tracker_id}, Class: {model.model.names[class_id]}, Confidence: {confidence:.2f}, Box: {list(xyxy)}\n")
            id = tracker_id
    
            cx = int((xyxy[0] + xyxy[2]) / 2)  # Calculate center x of the box
            cy = int((xyxy[1] + xyxy[3]) / 2)  # Calculate center y of the box
    
            # Draw the center for every bounding box unconditionally:
            cv2.circle(frame, (int(cx), int(cy)), 3, (0, 100, 255), -1)                  
    
            last_pos = id_to_last_position.get(id, (None, None))
    
            # Determine side (could be based on cx, or other logic)
            side = 'side1' if cx < frame.shape[1] // 2 else 'side2'
    
            # Call the counting function with the appropriate side settings
            count_vehicles(side, id,class_id, cx, cy, last_pos)
   # Assuming side counts are updated at this point
    file1.write(f"Frame {index} Side1 Counts: Total Up: {side_settings['side1']['total_count_up']}, Total Down: {side_settings['side1']['total_count_down']}\n")
    file1.write(f"Frame {index} Side2 Counts: Total Up: {side_settings['side2']['total_count_up']}, Total Down: {side_settings['side2']['total_count_down']}\n")
    # Append class-specific counts to the file
    for side in ['side1', 'side2']:
        settings = side_settings[side]
        for class_id in [2, 3, 5, 7]:
            file1.write(f"Frame {index} - {side} - Class {class_id} Counts: Up: {settings['counts'][class_id]['up']}, Down: {settings['counts'][class_id]['down']}\n")
    
    
    # Log or print the current number of vehicles between lines for each side
            file1.write(f"Frame {index} - Side1 Current Vehicles: {side_settings['side1']['current_vehicles']}\n")
            file1.write(f"Frame {index} - Side2 Current Vehicles: {side_settings['side2']['current_vehicles']}\n")
        
    
    
    
    line_coords_side1 = [((642, 120), (409, 80)), ((687, 242), (390, 228))]
    text_coords_side1 = [
        (f"Side1: Total Up: {side_settings['side1']['total_count_up']}", (180, 370)),
        (f"Side1: Total Down: {side_settings['side1']['total_count_down']}", (180, 390))
    ]
    draw_side_info(frame, 'side1', line_coords_side1, text_coords_side1)

    line_coords_side2 = [((361, 35), (168, 22)), ((260, 330), (1, 323))]
    text_coords_side2 = [
        (f"Side2: Total Up: {side_settings['side2']['total_count_up']}", (370, 320)),
        (f"Side2: Total Down: {side_settings['side2']['total_count_down']}", (370, 340))
    ]
    draw_side_info(frame, 'side2', line_coords_side2, text_coords_side2)

    
    labels = [
        f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for xyxy, confidence, class_id, tracker_id in zip(detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id)
    ]


    #Uses the BoxAnnotator to draw bounding boxes and labels on a copy of the current frame. 
    return annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)


#This function manages the streaming and processing of video from an RTSP URL.
def process_rtsp_stream(rtsp_url: str):
    
    #Opens the RTSP video stream for processing.
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        raise Exception(f"Could not open RTSP stream: {rtsp_url}")
    
    #Continuously reads frames from the video stream until no more frames are available
    index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        #Sends the RGB frame to the callback function for processing (detection and tracking), then converts the annotated frame back to BGR for display.
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_frame = callback(frame, index,result_file,result_file1)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        
        cv2.imshow('RTSP Stream', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        #Increments the frame index with each loop iteration.
        index += 1
    
    cap.release()
    cv2.destroyAllWindows()
    result_file.close()  # Close the file here after the stream is finished

rtsp_url = 'rtsp://.:cam@1234@127.0.0.1:45254/cam/realmonitor?channel=1&subtype=1'
process_rtsp_stream(rtsp_url)