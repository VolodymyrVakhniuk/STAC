import math, cv2
from time import time
import numpy as np
from sklearn import svm
from matplotlib import pyplot as plt
import General as general
import RegressionFilter

DET_PATH =  general.DET_PATH
DET_FILENAMES = general.DET_FILENAMES

TIME_WINDOW = general.TIME_WINDOW

cameras = general.CAMERAS
cameras_shape = general.cameras_shape


def prepare_svm_data(reid_hash_map, source_cam, destination_cam, time_window):
    svm_data, svm_label, data_id = [], [], []
    
    for t, obj in reid_hash_map[source_cam]:
        if t < time_window[0] or t >= time_window[1]:
            continue
        svm_data.append(reid_hash_map[source_cam][(t, obj)])
        data_id.append((t, obj))
        label_data = 0
        for off_set in range(1):
            if (t + off_set, obj) in reid_hash_map[destination_cam]:
                label_data = 1
        svm_label.append(label_data)

    print(source_cam, destination_cam, sum(svm_label))
    return np.array(svm_data), np.array(svm_label), data_id


def get_SVM_HashMap(gamma=2e-3, res_thres=2.0):

    reid_hash_map = RegressionFilter.get_Regression_Hashmap(res_thres=res_thres)
    exist_max_oid = -1

    for cam_name in reid_hash_map:
        for _, obj in reid_hash_map[cam_name]:
            exist_max_oid = max(exist_max_oid, obj)
    
    # Logic to add some extra bbox from yolo detection
    for cam_idx, cam_name in enumerate(cameras):
        time_to_bbox = {}
        for frame_id, oid in reid_hash_map[cam_name]:
            if frame_id not in time_to_bbox:
                time_to_bbox[frame_id] = [reid_hash_map[cam_name][(frame_id, oid)]]
            else:
                time_to_bbox[frame_id].append(reid_hash_map[cam_name][(frame_id, oid)])

        for line in open(DET_PATH + DET_FILENAMES[cam_idx]).readlines():
            frame_id, _, left, top, width, height, confidence, _, _, _ = [float(each) for each in line.split(',')]
            if confidence < 0.05: continue
            if left < 50 or top < 50 or left + width + 50 > cameras_shape[cam_name][1] or \
                top + height + 50 > cameras_shape[cam_name][0]: continue
            frame_id, left, top, width, height = round(frame_id), round(left), round(top), round(width), round(height)

            add_this = True
            if frame_id in time_to_bbox:
                for exist_bbox in time_to_bbox[frame_id]:
                    if general.IoU(exist_bbox, (left, top, width, height)) > 0.1:
                        add_this = False
                        break
            if add_this:
                exist_max_oid += 1
                reid_hash_map[cam_name][(frame_id, exist_max_oid)] = (left, top, width, height)


    outlier_dict = {cam: [] for cam in cameras}
    unique_dict = {cam: [] for cam in cameras}

    print("SVM function gamma input ", gamma)

    for source_cam in cameras:
        source_unique = set(reid_hash_map[source_cam].keys())
        source_outlier = set()
        for destination_cam in cameras:
            if source_cam == destination_cam:
                continue
            svm_data, svm_label, data_id = prepare_svm_data(reid_hash_map, source_cam, destination_cam, TIME_WINDOW)       
            false_pos = list(np.where(svm_label == 0)[0])
            source_unique = source_unique.intersection(set([data_id[i] for i in false_pos]))

            clf = svm.SVC(kernel='rbf', class_weight='balanced', gamma=gamma)  # change to 2e-5 in overall setup
            clf.fit(svm_data, svm_label)
            y_pred = clf.predict(svm_data)

            false_negative_pos = list(np.where((svm_label - y_pred) == -1)[0])
            if len(false_negative_pos) == 0: continue
            tmp_outlier = set([data_id[i] for i in false_negative_pos])
            source_outlier = source_outlier.union(tmp_outlier)    
        outlier_dict[source_cam] = source_unique.intersection(source_outlier)
        unique_dict[source_cam] = source_unique

    outlier_num = 0

    Multi_Hashmap = {}
    for cam in cameras:
        for key in outlier_dict[cam]:
            del reid_hash_map[cam][key]
            outlier_num += 1
        Multi_Hashmap[cam] = reid_hash_map[cam]

    print('SVM Outliers Number', outlier_num)

    return Multi_Hashmap, reid_hash_map,  outlier_dict, unique_dict
