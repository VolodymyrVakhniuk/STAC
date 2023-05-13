# Appending current yolo detections to a .txt file
def save_yolo_detections(filtered_detections_across_frames, frame_idx):
    # Format: frame, -1, left, top, width, height, conf, -1, -1, -1

    with open("YoloDetections/camera_8_yolo.txt", "a") as f:
        for det in filtered_detections_across_frames[0]:  
            left, top = det[0], det[1]
            width, height = det[2] - left, det[3] - top
            conf = det[-1]
            f.write(f"{frame_idx},-1,{left},{top},{width},{height},{conf},-1,-1,-1\n")
    
    with open("YoloDetections/camera_9_yolo.txt", "a") as f:
        for det in filtered_detections_across_frames[1]:  
            left, top = det[0], det[1]
            width, height = det[2] - left, det[3] - top
            conf = det[-1]
            f.write(f"{frame_idx},-1,{left},{top},{width},{height},{conf},-1,-1,-1\n")


# Appending current person ReID results to a .txt file
# frame_idx_t stands for the current timestep where the current frames are extracted
def save_reid_results(bboxes_across_frames, frame_idx_t):
    # Format: cid, oid, frame_idx, startX, startY, Width, Height, -1, -1
    # Format: 2 1586 1 883 190 140 139 -1 -1

    with open("ReIDResults/reid.txt", "a") as f:

        for bbox in bboxes_across_frames[0]:
            if bbox != 0:
                cid = 8
                startX, startY = bbox[0], bbox[1]
                Width, Height = bbox[2] - startX, bbox[3] - startY
                oid = bbox[5]

                f.write(f"{cid} {oid} {frame_idx_t} {startX} {startY} {Width} {Height} -1 -1\n")

        for bbox in bboxes_across_frames[1]:
            if bbox != 0:
                cid = 9
                startX, startY = bbox[0], bbox[1]
                Width, Height = bbox[2] - startX, bbox[3] - startY
                oid = bbox[5]

                f.write(f"{cid} {oid} {frame_idx_t} {startX} {startY} {Width} {Height} -1 -1\n")
