import os, cv2
from os.path import dirname
import numpy as np
import General as general
import subprocess
import shutil, time
from PIL import Image

def plot_frame_w_nouse_tiles(cam_name, frame_id, no_use_list, filename):
    base_frame = general.get_frame(cam_name, frame_id)
    f_width = general.cameras_shape[cam_name][1]
    t_height, t_width = general.tile_height, general.tile_width
    n_row = f_width // t_width
    for tile in no_use_list:
        left, top = (tile % n_row) * t_width, (tile // n_row) * t_height
        cv2.rectangle(base_frame, (left, top), (left + t_width, top + t_height), (0, 0, 0), -1)
    # cv2.imwrite(filename, base_frame)
    cv2.imshow("plot_frame_w_nouse_tiles", base_frame)
    cv2.waitKey()


def plot_ROI_mask(cam_name, no_use_list, filename):
    f_height, f_width = general.cameras_shape[cam_name][0], general.cameras_shape[cam_name][1]
    base_frame = np.ones((f_height, f_width, 3), dtype=np.uint8) * 255
    t_height, t_width = general.tile_height, general.tile_width
    n_row = f_width // t_width
    for tile in no_use_list:
        left, top = (tile % n_row) * t_width, (tile // n_row) * t_height
        cv2.rectangle(base_frame, (left, top), (left + t_width, top + t_height), (0, 0, 0), -1)
    # cv2.imwrite(filename, base_frame)
    cv2.imshow("plot_ROI_mask", base_frame)
    cv2.waitKey()

    
def generate_zero_padding_video(cam_name, no_use_list):
    pwd = os.getcwd()
    dir_name = os.path.join(pwd, cam_name + "_zero_pad", "")
    os.mkdir(dir_name)
    for i in range(300):
        image_path = dir_name + f'{i+1:04}'+ ".png"
        plot_frame_w_nouse_tiles(cam_name, i+1, no_use_list, image_path)

    outfile = cam_name + "_zero_pad" + ".mp4"

    encoding_result = subprocess.run(["ffmpeg", "-r", "10", "-f", "image2",
                                          "-s", "1920x1080", "-i", f"{dir_name}/%04d.png", 
                                          "-vcodec", "libx264", "-start_number", "1",
                                          "-pix_fmt", "yuv420p", "-crf", "23", outfile],
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE,
                                         universal_newlines=True)
    shutil.rmtree(dir_name)

    return encoding_result

def generate_org_video(cam_name):
    pwd = os.getcwd()
    dir_name = os.path.join(pwd, cam_name + "_org", "")
    os.mkdir(dir_name)
    for i in range(300):
        base_frame = general.get_frame(cam_name, i)
        image_path = dir_name + f'{i+1:04}'+ ".png"
        print(image_path)
        cv2.imwrite(image_path, base_frame)

    outfile = cam_name + "_org" + ".mp4"

    subprocess.run(["ffmpeg", "-r", "10", "-f", "image2",
                            "-s", "1920x1080", "-i", f"{dir_name}/%04d.png", 
                            "-vcodec", "libx264", "-start_number", "1",
                            "-pix_fmt", "yuv420p", "-crf", "23", outfile],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True)
    shutil.rmtree(dir_name)


def plot_gt_detected_obj(cam_name):
    vid = cv2.VideoCapture(general.WORKSAPCE + general.DATA_PATH + cam_name + '/' + 'vdo.avi')
    detections = {}
    for line in open(general.WORKSAPCE + general.DATA_PATH + cam_name + '/' + general.GT_PATH).readlines():
        frame_id, _, left, top, width, height, _, _, _, _ = [int(each) for each in line.split(',')]
        if frame_id not in detections:
            detections[frame_id] = [(left, top, width, height)]
        else:
            detections[frame_id].append((left, top, width, height))

    frame_count = 1
    while True:
        return_value, frame = vid.read()
        if not return_value:
            raise ValueError("No image!")
        if frame_count in detections:
            for left, top, width, height in detections[frame_count]:
                cv2.rectangle(frame, (left, top), (left + width, top + height), (255, 0, 0), 3)
        frame_count += 1
        time.sleep(0.09)
        
        cv2.imshow("result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
