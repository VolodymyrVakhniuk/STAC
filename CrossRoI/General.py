import math, cv2

# Path/Names to AIC23 dataset
DATASET_PATH = "./AIC23_Track1_MTMC_Tracking"
DATA_PATH = "/train/S002/"
SCENE_NAME = 'S002'
GT_FILENAME = "label.txt"

# Path/Names to Yolo detections
DET_PATH =  "./YoloDetections/"
DET_FILENAMES = ["camera_8_yolo.txt", "camera_9_yolo.txt"]

# Path/Names to ReID detections
REID_PATH = "./ReID/"
REID_FILENAME = "reid_r.txt"

# Other global variables
CAMERAS = ["c008", "c009"]
TIME_WINDOW = [0, 10]

# get the k-th frame of a certain camera
def get_frame(cam_name, frame_id):
	cap = cv2.VideoCapture(DATASET_PATH + DATA_PATH + cam_name + '/' + 'video.mp4')
	cap.set(1, frame_id)
	_, frame = cap.read()
	return frame

# Accounting for the fact that different cameras may have different resolutions
cameras_shape = {camera: get_frame(camera, 0).shape[:2] for camera in CAMERAS}
print(f"Shape of frame: {cameras_shape}")

# Induces 8x8 grid on top of each frame (in the case resolution is 1080x1920).
tile_height, tile_width = 270, 240
cam_to_tshape = {cam: (math.ceil(cameras_shape[cam][0] / tile_height), math.ceil(cameras_shape[cam][1] / tile_width)) for cam in CAMERAS}

# IoU score of two given rectanglar bbox
def IoU(rect1,rect2):
	x1, y1, w1, h1 = rect1
	x2, y2, w2, h2 = rect2

	inter_w = (w1 + w2) - (max(x1 + w1, x2 + w2) - min(x1, x2))
	inter_h = (h1 + h2) - (max(y1 + h1, y2 + h2) - min(y1, y2))

	if inter_h<=0 or inter_w <= 0:
		return 0
	inter = inter_w * inter_h
	union = w1 * h1 + w2 * h2 - inter
	return inter / union

# Used for filtering: (gamma, res_thres)
experiment_setting = [(1e-6, 1.0), (5e-6, 1.0), (1e-5, 1.0), (5e-5, 1.0), (1e-4, 1.0), \
					  (2e-5, 0.01), (2e-5, 0.05), (2e-5, 0.1), (2e-5, 1.0),(2e-5, 10.0), \
					  (100, 200)]

experiment_subdirs = ['1e-06_1.0', '5e-06_1.0', '1e-05_1.0', '5e-05_1.0', '1e-04_1.0', \
					  '2e-05_0.01', '2e-05_0.05', '2e-05_0.1',  '2e-05_1.0', '2e-05_10.0',\
					  'nofilter' ]
