import math, cv2
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from statsmodels import robust
import General as general

REID_PATH = general.REID_PATH
REID_FILENAME = general.REID_FILENAME
TIME_WINDOW = general.TIME_WINDOW

cameras = general.CAMERAS

def get_Regression_Hashmap(res_thres=2):

    reid_hash_map = {cam: {} for cam in cameras}
    exist_max_oid = -1

    for line in open(REID_PATH + REID_FILENAME).readlines():
        cid, oid, frame_id, left, top, width, height = [int(line.split(' ')[i]) for i in range(7)]
        exist_max_oid = max(exist_max_oid, oid)

        # Our cameras have id 8 and 9 => we subtract 8 to convert those to indices 0 and 1:
        reid_hash_map[cameras[cid - 8]][(frame_id, oid)] = (left, top, width, height)

    def prepare_regression_data(reid_hash_map, source_cam, destination_cam, time_window):
        source_data, destination_data, source_data_id, destination_data_id = [], [], [], []

        for t, obj in reid_hash_map[source_cam]:
            if t < time_window[0] or t >= time_window[1]:
                continue
            if (t, obj) not in reid_hash_map[destination_cam]:
                continue
            source_data.append(list(reid_hash_map[source_cam][(t, obj)]))
            destination_data.append(list(reid_hash_map[destination_cam][(t, obj)]))
            source_data_id.append((t, obj))
            destination_data_id.append((t, obj))

        return np.array(source_data), np.array(destination_data), source_data_id, destination_data_id


    def frame_obj_to_cameras(reid_hash_map, cameras, time_window):
        fo_to_cams = {}

        for cam in cameras:
            for t, obj in reid_hash_map[cam]:
                if t < time_window[0] or t >= time_window[1]:
                    continue
                if (t, obj) not in fo_to_cams:
                    fo_to_cams[(t, obj)] = [cam]
                else:
                    fo_to_cams[(t, obj)].append(cam)

        return fo_to_cams

    fo_to_cams = frame_obj_to_cameras(reid_hash_map, cameras, TIME_WINDOW)
    cam_to_outliers = {}

    for source_cam in cameras:
        outlier = set() 
        for destination_cam in cameras:
            if source_cam == destination_cam:
                continue
            source_data, destination_data, source_data_id, destination_data_id = \
                prepare_regression_data(reid_hash_map, source_cam, destination_cam, TIME_WINDOW)

            degree = 4 if source_data.shape[0] > 80 else 3
            if source_data.shape[0] > 120: degree = 4
            if source_data.shape[0] > 150: degree = 5
            if source_data.shape[0] > 220: degree = 6

            y_mad = np.linalg.norm(robust.mad(destination_data, axis=0), 2)
            regr = make_pipeline(PolynomialFeatures(degree),linear_model.RANSACRegressor(residual_threshold = res_thres * y_mad))
            regr.fit(source_data, destination_data)
            inlier_mask = regr[-1].inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)

            tmp_outlier_pos = list(np.where(outlier_mask == True)[0])
            outlier = outlier.union(set([source_data_id[i] for i in tmp_outlier_pos]))

        cam_to_outliers[source_cam] = outlier

    for cam in cameras:
        for frame, obj in cam_to_outliers[cam]:
            if (frame, obj) in fo_to_cams:
                fo_to_cams[(frame, obj)].remove(cam)

    true_outliers = [key for key in fo_to_cams.keys() if len(fo_to_cams[key]) == 0]

    print('Regression max obj id', exist_max_oid)
    print('Regression Outlier Number', len(true_outliers))

    ## clean reid_hashmap with true_outlier
    for cam in cameras:
        for frame, obj in true_outliers:
            if (frame, obj) not in reid_hash_map[cam]:
                continue
            bbox = reid_hash_map[cam][(frame, obj)]
            del reid_hash_map[cam][(frame, obj)]
            exist_max_oid += 1
            reid_hash_map[cam][(frame, exist_max_oid)] = bbox
            

    Multi_Hashmap = reid_hash_map
    return Multi_Hashmap