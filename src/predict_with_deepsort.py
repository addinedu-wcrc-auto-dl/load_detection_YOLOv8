from ultralytics import YOLO
import time
import cv2
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from deep_sort.deep_sort import DeepSort
from deep_sort.sort.tracker import Tracker

# Load a model
model = YOLO("yolov8l-seg_best.pt")  # load a pretrained model (recommended for training)
# model = YOLO("best.pt")

deep_sort_weights = os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + '/deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, max_age=70)

# Define the video path
video_path = '20231118_185609.mp4'

cap = cv2.VideoCapture(video_path)

# Get the video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
output_path = 'output.avi'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

unique_track_ids = set()

i = 0
counter, fps, elapsed = 0, 0, 0
start_time = time.perf_counter()

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        
        og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = og_frame.copy()

        results = model(frame)

        # class_names = {0: 'flatness_A', 1: 'flatness_B', 2: 'flatness_C', 3: 'flatness_D', 4: 'flatness_E', 5: 'walkway_paved', 6: 'walkway_block', 7: 'paved_state_broken', 8: 'paved_state_normal', 9: 'block_state_broken', 10: 'block_state_normal', 11: 'block_kind_bad', 12: 'block_kind_good', 13: 'outcurb_rectangle', 14: 'outcurb_slide', 15: 'outcurb_slide_broken', 16: 'outcurb_rectangle_broken', 17: 'restspace', 18: 'sidegap_in', 19: 'sidegap_out', 20: 'sewer_cross', 21: 'sewer_line', 22: 'brailleblock_dot', 23: 'brailleblock_line', 24: 'brailleblock_dot_broken', 25: 'brailleblock_line_broken', 26: 'continuity_tree', 27: 'continuity_manhole', 28: 'ramp_yes', 29: 'ramp_no', 30: 'bicycleroad_broken', 31: 'bicycleroad_normal', 32: 'planecrosswalk_broken', 33: 'planecrosswalk_normal', 34: 'steepramp', 35: 'bump_slow', 36: 'bump_zigzag', 37: 'weed', 38: 'floor_normal', 39: 'floor_broken', 40: 'flowerbed', 41: 'parkspace', 42: 'tierbump', 43: 'stone', 44: 'enterrail', 45: 'stair_normal', 46: 'stair_broken', 47: 'wall', 48: 'window_sliding', 49: 'window_casement', 50: 'pillar', 51: 'lift', 52: 'door_normal', 53: 'door_rotation', 54: 'lift_door', 55: 'resting_place_roof', 56: 'reception_desk', 57: 'protect_wall_protective', 58: 'protect_wall_guardrail', 59: 'protect_wall_kickplate', 60: 'handle_vertical', 61: 'handle_lever', 62: 'handle_circular', 63: 'lift_button_normal', 64: 'lift_button_openarea', 65: 'lift_button_layer', 66: 'lift_button_emergency', 67: 'direction_sign_left', 68: 'direction_sign_right', 69: 'direction_sign_straight', 70: 'direction_sign_exit', 71: 'sign_disabled_toilet', 72: 'sign_disabled_parking', 73: 'sign_disabled_elevator', 74: 'sign_disabled_callbell', 75: 'sign_disabled_icon', 76: 'sign_disabled_ramp', 77: 'braille_sign', 78: 'chair_multi', 79: 'chair_one', 80: 'chair_circular', 81: 'chair_back', 82: 'chair_handle', 83: 'number_ticket_machine', 84: 'beverage_vending_machine', 85: 'beverage_desk', 86: 'trash_can', 87: 'mailbox', 88: 'fireshutter'}
        
        # 윤님 작업물 class
        class_names = {0: "flatness_A",   1: "flatness_B",   2: "flatness_C",   3: "flatness_D",   4: "flatness_E",   5: "walkway_paved",   6: "walkway_block",   7: "paved_state_broken",   8: "paved_state_normal",   9: "block_state_broken",   10: "block_state_normal",   11: "block_kind_bad",   12: "block_kind_good",   13: "outcurb_rectangle",   14: "outcurb_slide",   15: "outcurb_rectangle_broken",   16: "restspace",   17: "sidegap_in",   18: "sidegap_out",   19: "sewer_cross",   20: "sewer_line",   21: "brailleblock_dot",   22: "brailleblock_line",   23: "brailleblock_dot_broken",   24: "brailleblock_line_broken",   25: "continuity_tree",   26: "continuity_manhole",   27: "ramp_yes",   28: "ramp_no",   29: "bicycleroad_broken",   30: "bicycleroad_normal",   31: "planecrosswalk_broken",   32: "planecrosswalk_normal",   33: "steepramp",   34: "bump_slow",   35: "weed",   36: "floor_normal",   37: "floor_broken",   38: "flowerbed",   39: "parkspace",   40: "tierbump",   41: "stone",   42: "enterrail",   43: "stair_normal",   44: "stair_broken",   45: "wall",   46: "window_sliding",   47: "window_casement",   48: "pillar",   49: "lift",   50: "door_normal",   51: "lift_door",   52: "resting_place_roof",   53: "reception_desk",   54: "protect_wall_protective",   55: "protect_wall_guardrail",   56: "protect_wall_kickplate",   57: "handle_vertical",   58: "handle_lever",   59: "handle_circular",   60: "lift_button_normal",   61: "lift_button_openarea",   62: "lift_button_layer",   63: "lift_button_emergency",   64: "direction_sign_left",   65: "direction_sign_right",   66: "direction_sign_straight",   67: "direction_sign_exit",   68: "sign_disabled_toilet",   69: "sign_disabled_parking",   70: "sign_disabled_elevator",   71: "sign_disabled_callbell",   72: "sign_disabled_icon",   73: "braille_sign",   74: "chair_multi",   75: "chair_one",   76: "chair_circular",   77: "chair_back",   78: "chair_handle",   79: "number_ticker_machine",   80: "beverage_vending_machine",   81: "beverage_desk",   82: "trash_can",   83: "mailbox",   84: "door_rotation",   85: "fireshutter",   86: "outcurb_slide_broken",   87: "bump_zigzag",   88: "sign_disabled_ramp",   89: "number_ticket_machine"}

        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            probs = result.probs  # Class probabilities for classification outputs
            cls = boxes.cls.tolist()  # Convert tensor to list
            xyxy = boxes.xyxy
            conf = boxes.conf
            xywh = boxes.xywh  # box with xywh format, (N, 4)
            for i, class_index in enumerate(cls):
                class_name = class_names[int(class_index)]
                print("Class:", class_name)     
                
                
                # # for checking without DeepSort
                # # Set color values for red, blue, and green
                # red_color = (0, 0, 255)  # (B, G, R)
                # blue_color = (255, 0, 0)  # (B, G, R)
                # green_color = (0, 255, 0)  # (B, G, R)

                # # Determine color based on track_id
                # color_id = class_index % 3
                # if color_id == 0:
                #     color = red_color
                # elif color_id == 1:
                #     color = blue_color
                # else:
                #     color = green_color
                
                # cv2.rectangle(og_frame, (int(xyxy[i][0]), int(xyxy[i][1])), (int(xyxy[i][2]), int(xyxy[i][3])), (255, 0, 0), 2)

                # text_color = (0, 0, 0)  # Black color for text
                # cv2.putText(og_frame, f"{class_name}", (int(xyxy[i][0]) + 10, int(xyxy[i][1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        pred_cls = np.array(cls)
        conf = conf.detach().cpu().numpy()
        xyxy = xyxy.detach().cpu().numpy()
        bboxes_xywh = xywh
        bboxes_xywh = xywh.cpu().numpy()
        bboxes_xywh = np.array(bboxes_xywh, dtype=float)
        
        if len(bboxes_xywh) <= 0:
            continue
        
        tracks = tracker.update(bboxes_xywh, conf, og_frame)
        
        for track in tracker.tracker.tracks:
            track_id = track.track_id
            hits = track.hits
            x1, y1, x2, y2 = track.to_tlbr()  # Get bounding box coordinates in (x1, y1, x2, y2) format
            w = x2 - x1  # Calculate width
            h = y2 - y1  # Calculate height

            # Set color values for red, blue, and green
            red_color = (0, 0, 255)  # (B, G, R)
            blue_color = (255, 0, 0)  # (B, G, R)
            green_color = (0, 255, 0)  # (B, G, R)

            # Determine color based on track_id
            color_id = track_id % 3
            if color_id == 0:
                color = red_color
            elif color_id == 1:
                color = blue_color
            else:
                color = green_color

            cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

            text_color = (0, 0, 0)  # Black color for text
            cv2.putText(og_frame, f"{class_name}-{track_id}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

            # Add the track_id to the set of unique track IDs
            unique_track_ids.add(track_id)

        # Update FPS and place on frame
        current_time = time.perf_counter()
        elapsed = (current_time - start_time)
        counter += 1
        if elapsed > 1:
            fps = counter / elapsed
            counter = 0
            start_time = current_time

        # Write the frame to the output video file
        out.write(cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))

        # Show the frame
        cv2.imshow("Video", cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()