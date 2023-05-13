gt_filenames = ['label_1.txt', 'label_2.txt']

# IoU score of two given rectanglar bbox
def IoU(rect1, rect2):
	x1, y1, w1, h1 = rect1
	x2, y2, w2, h2 = rect2

	inter_w = (w1 + w2) - (max(x1 + w1, x2 + w2) - min(x1, x2))
	inter_h = (h1 + h2) - (max(y1 + h1, y2 + h2) - min(y1, y2))

	if inter_h<=0 or inter_w <= 0:
		return 0
	inter = inter_w * inter_h
	union = w1 * h1 + w2 * h2 - inter
	return inter / union

# Getting ground truths detection boxes in given frame (current_frame_idx) 
# of a given camera (incorporated into bboxes_gt_filepath)
def get_gt_boxes_fc(bboxes_gt_filepath, current_frame_idx):

	# Getting gt bboxes
	frame_gt_bboxes = []
	with open(bboxes_gt_filepath, 'r') as f:
		for line in f:
			elements = line.strip().split(',')

			frame_idx = int(elements[0])
	
			if frame_idx == current_frame_idx:
				# Format: 2,9,130,322,46,77,1,-1,-1,-1
				startX, startY = int(elements[2]), int(elements[3])
				endX, endY = int(elements[4]), int(elements[5])

				frame_gt_bboxes.append([startX, startY, endX, endY])

			elif int(elements[0]) > current_frame_idx:
				break
	
	return frame_gt_bboxes


# Getting detection boxes in given frame of a given camera (camera_idx)
def get_det_boxes_fc(bboxes_in_frame):
	frame_det_bboxes = []
	for bbox in bboxes_in_frame:
		if bbox != 0:
			startX, startY, endX, endY = bbox[0], bbox[1], bbox[2], bbox[3]
			frame_det_bboxes.append([startX, startY, endX - startX, endY - startY])

	return frame_det_bboxes


def compute_accuracy_frame(frame_gt_bboxes, frame_det_bboxes):
	num_correct_detections = 0

	for gt_bbox in frame_gt_bboxes:
		IoU_scores = []

		# Detected boxes and ground truth boxes don't come in 1-1 correspondence =>
		# We select the detection box that matches the given gt box the best.
		for det_bbox in frame_det_bboxes:
			IoU_scores.append(IoU(det_bbox, gt_bbox))

		# If the best match has high IoU score then the detected box is accurate.
		if max(IoU_scores) >= 0.7:
			num_correct_detections += 1

	num_gt_total = len(frame_gt_bboxes)
	frame_accuracy = num_correct_detections / num_gt_total

	return frame_accuracy, num_correct_detections, num_gt_total


def get_accuracy_score(current_frame_index, bboxes_across_frames):	

	# Getting gt bboxes
	frame_1_gt_bboxes = get_gt_boxes_fc('demo_videos/label_1.txt', current_frame_index)
	frame_2_gt_bboxes = get_gt_boxes_fc('demo_videos/label_2.txt', current_frame_index)

	# Getting det bboxes, where 0, 1 are camera indices
	frame_1_det_bboxes = get_det_boxes_fc(bboxes_across_frames[0])
	frame_2_det_bboxes = get_det_boxes_fc(bboxes_across_frames[1])

	# Computing accuracy
	accuracy_frame_1, num_correct_dets_frame_1, num_gt_total_frame_1 = compute_accuracy_frame(frame_1_gt_bboxes, frame_1_det_bboxes)
	accuracy_frame_2, num_correct_dets_frame_2, num_gt_total_frame_2 = compute_accuracy_frame(frame_2_gt_bboxes, frame_2_det_bboxes)

	num_correct_dets_across_frames = num_correct_dets_frame_1 + num_correct_dets_frame_2
	num_gt_total_across_frames = num_gt_total_frame_1 + num_gt_total_frame_2
	accuracy_across_frames = num_correct_dets_across_frames / num_gt_total_across_frames

	# Second and third return variable needed because we will modify the accuracy "eyeballing" the 
	# ID detections
	return accuracy_across_frames, num_correct_dets_across_frames, num_gt_total_across_frames