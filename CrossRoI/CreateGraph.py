import math, cv2
from os import confstr_names
import numpy as np
from numpy.lib.npyio import _savez_compressed_dispatcher
import General as general
import Optimizer
import Visualizer
import MergeTile
import SVMFilter
import RegressionFilter

# environment & macro definations
DATASET_PATH = general.DATASET_PATH
DATA_PATH = general.DATA_PATH
GT_FILENAME = general.GT_FILENAME

REID_PATH = general.REID_PATH
REID_FILENAME = general.REID_FILENAME

TIME_WINDOW = general.TIME_WINDOW

cameras = general.CAMERAS
cameras_shape = general.cameras_shape
tile_height, tile_width = general.tile_height, general.tile_width
cam_to_tshape = general.cam_to_tshape

DO_FILTERING = False


# Given a bbox, return the tiles which cover that bbox. 
# Note: cam_name is needed because different cameras may have different resolution
def bbox_to_tiles(bbox, cam_name):
	f_height, f_width = cameras_shape[cam_name]
	row_n =  f_width // tile_width
	
	left, top, width, height = bbox
	first_tile = (top // tile_height) * row_n + (left // tile_width)
	last_tile = ((top + height) // tile_height) * row_n + ((left + width) // tile_width)
	result = []
	for tile in range(first_tile, last_tile + 1):
		if not (left // tile_width) <= tile % row_n <= ((left + width) // tile_width):
			continue
		result.append(tile)

	return result


# Covernts gt results (in gt.txt file) to camera data hashmap
# hashmap := (frame, obj_id) -> (bbox)
# Note: Each camera has a corresponding gt_hashmap
def data_to_hashmap(cam_name):
	gt = open(DATASET_PATH + DATA_PATH + cam_name + '/' + GT_FILENAME).read().split('\n')[:-1]
	print(f"gt dir is: {DATASET_PATH + DATA_PATH + cam_name + '/' + GT_FILENAME}")

	gt_hashmap = {(int(each.split(',')[0]), int(each.split(',')[1])): \
				   tuple([int(i) for i in each.split(',')[2:6]]) for each in gt}
	return gt_hashmap


# Generate the hashmap containing unified cameras data:
# multi_sync_hashmap := (cam) -> hashmap_tiles
# where hashmap_tiles is same as hashmap but uses tiles instead of bboxes.
def multi_cam_hashmap(cam_list, time_window, gt_multi_hashmap=None):

    # In case filtering has not been done:
    if gt_multi_hashmap is None:
        gt_multi_hashmap = {cam: data_to_hashmap(cam) for cam in cam_list}

    # Convert bbox into tile list formation
    multi_sync_hashmap = {}
    time_to_obj = {t: set() for t in range(time_window[0], time_window[1])}

    for cam in gt_multi_hashmap:
        hashmap_tiles = {}

        # Note: t stands for frame
        for t, obj in gt_multi_hashmap[cam]:
    
            # Throwing away all frames outside specified time window
            # Note: time is measured in FRAMES (not seconds, minutes, etc).
            if t < time_window[0] or t >= time_window[1]:
                continue
        
            # Converting bboxes to tiles
            bbox = gt_multi_hashmap[cam][(t, obj)]
            hashmap_tiles[(t, obj)] = (bbox_to_tiles(bbox, cam))
            time_to_obj[t].add(obj)
        
        # Storing hashmap_tiles for a corresponding camera
        multi_sync_hashmap[cam] = hashmap_tiles
    
    return multi_sync_hashmap, time_to_obj


if __name__ == "__main__":
	experiment_setting = general.experiment_setting

	for i, (gamma, res_thresh) in enumerate(experiment_setting):
		
		filtered_hashmap = None
		if DO_FILTERING == True:
			filtered_hashmap, _, _, _ = SVMFilter.get_SVM_HashMap(gamma=gamma, res_thres=res_thresh)

		camera_used_blocks = Optimizer.optimization_solver(cameras, cam_to_tshape, TIME_WINDOW, \
		 												gt_multi_hashmap=filtered_hashmap)		

		camera_nouse_blocks = {cam: set(list(range(cam_to_tshape[cam][0] * cam_to_tshape[cam][1]))) \
								- set(camera_used_blocks[cam]) for cam in cameras}		

		for cam in cameras:
			Visualizer.plot_frame_w_nouse_tiles(cam, 20, camera_nouse_blocks[cam], "")
									 