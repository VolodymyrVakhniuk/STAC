import os
from init import *
from eval_util import *
from save import *
import copy
import time

video_directory = 'demo_videos'
video_filenames = ['video_1.mp4', 'video_2.mp4']
gt_filenames = ['label_1.txt', 'label_2.txt']
STARTING_FRAME_IDX = 0


def get_list_of_videos():
	video_filepaths = [os.path.join(video_directory, video_filename) for video_filename in video_filenames]
	list_of_videos = [cv2.VideoCapture(video_filepath) for video_filepath in video_filepaths]
	list_of_video_fps = [video.get(cv2.CAP_PROP_FPS) for video in list_of_videos]
	return list_of_videos, list_of_video_fps


def extract_next_frames(list_of_videos):
	frames = []
	success = True
	for video in list_of_videos:
		ret, frame = video.read()
		if not ret:
			success = False
		frames.append(frame)
	return frames, success


def perform_yolo_detections(frames):
	# Converting frames
	color_converted_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

	# Do raw Yolo detections
	detections_across_frames = [detect(frame).cpu().detach().numpy() for frame in color_converted_frames]

	# Perform the filering
	filtered_detections_across_frames = [[] for i in range(len(frames))]
	cropped_people_across_frames = [[] for i in range(len(frames))]

	for frame_index, list_of_detections in enumerate(detections_across_frames):
		for detection in list_of_detections:
			x1, y1, x2, y2, confidence, class_label = detection
			startX, startY, endX, endY = int(x1), int(y1), int(x2), int(y2)
			class_name = class_names[int(class_label)]
			if confidence > 0.45 and class_name == 'person':
				filtered_detections_across_frames[frame_index].append([startX, startY, endX, endY, class_name, round(confidence, 2)])

				cropped_person = color_converted_frames[frame_index][startY:endY, startX:endX]
				cropped_people_across_frames[frame_index].append(cropped_person)

	return filtered_detections_across_frames, cropped_people_across_frames


def get_current_feature_vectors_across_frames(cropped_people_across_frames):
	current_feature_vectors_across_frames = []
	for cropped_people_in_frame in cropped_people_across_frames:
		current_feature_vector_in_frame = extractor(cropped_people_in_frame)
		current_feature_vectors_across_frames.append(current_feature_vector_in_frame)

	return current_feature_vectors_across_frames


def get_similarity_matrix(frame_one_features, frame_two_features):
	# Each frame features look like: [[512 x 1], [512 x 1], [512 x 1]...]
	similarity_matrix = 1 - torchreid.metrics.compute_distance_matrix(frame_one_features, frame_two_features,'cosine').cpu().detach().numpy()
	return similarity_matrix


def get_feature_comparisons(current_feature_vectors_across_frames, previous_feature_vectors_across_frames):

	num_curr_frames = len(current_feature_vectors_across_frames)

	# List: num_frames * num_frames, upper-triangular, 0s on the diagonal
	new_vs_new_feature_comparisons = [[0] * num_curr_frames for i in range(num_curr_frames)]
	new_vs_old_feature_comparisons = []

	
	for frame_index_i, frame_feature_vectors_i in enumerate(current_feature_vectors_across_frames):

		# Creating similarity matrices for each pair of current frames (across space)
		for frame_index_j, frame_feature_vectors_j in zip(range(frame_index_i + 1, num_curr_frames), current_feature_vectors_across_frames[frame_index_i + 1:]):
			similarity_matrix_ij = get_similarity_matrix(frame_feature_vectors_j, frame_feature_vectors_i)
			new_vs_new_feature_comparisons[frame_index_i][frame_index_j] = similarity_matrix_ij

		# Creating similarity matrices for temporal dimension
		new_frame_feature_vectors = frame_feature_vectors_i
		old_frame_index = frame_index_i
		old_frame_feature_vectors = previous_feature_vectors_across_frames[old_frame_index]
		similarity_matrix_new_vs_old = get_similarity_matrix(new_frame_feature_vectors, old_frame_feature_vectors)
		new_vs_old_feature_comparisons.append(similarity_matrix_new_vs_old)

	return new_vs_new_feature_comparisons, new_vs_old_feature_comparisons


# Across time
# Filetered detections are needed to extract the bounding boxes
def person_reid_through_time(new_vs_old_feature_comparisons,\
			      current_feature_vectors_across_frames,\
				  previous_feature_vectors_across_frames,\
				  filtered_detections_across_frames):
	
	num_curr_frames = len(current_feature_vectors_across_frames)
	bboxes_across_frames = [[0] * len(current_feature_vectors_across_frames[i]) for i in range(num_curr_frames)]

	for frame_idx, frame_similarity_matrix in enumerate(new_vs_old_feature_comparisons):

		detections_in_frame = filtered_detections_across_frames[frame_idx]
		current_feature_vectors_in_frame = current_feature_vectors_across_frames[frame_idx]
		previous_feature_vectors_in_frame = previous_feature_vectors_across_frames[frame_idx]
		
		query_id_used, gallery_id_used = [], []

		# Note: -np.sort(-np.concatenate(sim_res)) does sort from biggest to smallest.
		for sim in -np.sort(-np.concatenate(frame_similarity_matrix)):
			sim_ids = np.array(np.where(frame_similarity_matrix == sim)).T
			for query_id, gallery_id in sim_ids:
                
				startX, startY, endX, endY, class_name, conf = detections_in_frame[query_id]

				# If a person in current frame looks like a person in previous frame:
				if query_id not in query_id_used and gallery_id not in gallery_id_used and sim > 0.60:
                            
					bboxes_across_frames[frame_idx][query_id] = [startX, startY, endX, endY, class_name, gallery_id]
					
					# If an person in current frame looks too alike a person in previous frame, then update its feature vector:
					if sim > 0.65:
						previous_feature_vectors_in_frame[gallery_id] = current_feature_vectors_in_frame[query_id]

					# Making sure each person in current frame is matched to exactly one person in previous frame:
					query_id_used.append(query_id), gallery_id_used.append(gallery_id)

				# If a person in current frame does not look like any person in previous frame,
				# then the given person must have just appeared in this camera view => add new person for tracking
				elif query_id not in query_id_used and frame_similarity_matrix[query_id].max() < 0.55:
					previous_feature_vectors_in_frame = torch.cat([previous_feature_vectors_in_frame, torch.tensor([current_feature_vectors_in_frame[query_id].tolist()]).to(device)])
					query_id_used.append(query_id)

	# Adding offsets, aka making sure different people in different cameras have different IDs.
	for frame_idx in range(num_curr_frames):
		frame_bboxes = bboxes_across_frames[frame_idx]
		num_bboxes_in_given_frame = len(current_feature_vectors_across_frames[frame_idx])

		for bbox_idx in range(num_bboxes_in_given_frame):
			# frame_idx is the offset
			if frame_bboxes[bbox_idx] != 0:
				frame_bboxes[bbox_idx][-1] += frame_idx * 10
	
	return bboxes_across_frames


# Across space
def person_reid_through_frames(new_vs_new_feature_comparisons,\
			        current_feature_vectors_across_frames,\
					bboxes_across_frames):
	
	num_curr_frames = len(current_feature_vectors_across_frames)
	
	# Running feature comparisons across each pair of current frames
	for frame_index_i in range(num_curr_frames):
		for frame_index_j in range(frame_index_i + 1, num_curr_frames):

			query_id_used, gallery_id_used = [], []
			similarity_matrix_ij = new_vs_new_feature_comparisons[frame_index_i][frame_index_j]

			# Note: -np.sort(-np.concatenate(sim_res)) does sort from biggest to smallest.
			for sim in -np.sort(-np.concatenate(similarity_matrix_ij)):
				sim_ids = np.array(np.where(similarity_matrix_ij == sim)).T

				for query_id, gallery_id in sim_ids:
					
					if query_id not in query_id_used and gallery_id not in gallery_id_used and sim > 0.65:

						# If a person in frame_j looks like a person in frame_i then assign a person in frame_j
						# the ID of person in frame_i:
						if bboxes_across_frames[frame_index_i][gallery_id] != 0 and bboxes_across_frames[frame_index_j][query_id] != 0:
							object_id = bboxes_across_frames[frame_index_i][gallery_id][-1]
							bboxes_across_frames[frame_index_j][query_id][-1] = object_id

						# Making sure each person in frame_j is matched to exactly one person in frame_i:
						query_id_used.append(query_id), gallery_id_used.append(gallery_id)
		
	return bboxes_across_frames


### VISUALIZATION ROUTINE START ###
def create_cv2_bboxes_across_frames(bboxes_across_frames, frames):

	for frame_idx, frame in enumerate(frames):
		for bbox in bboxes_across_frames[frame_idx]:
			
			if bbox != 0:
				startX, startY, endX, endY = bbox[0], bbox[1], bbox[2], bbox[3]
				class_name, gallery_id = bbox[4], bbox[5]

				cv2.rectangle(frames[frame_idx], (startX, startY), (endX, endY), get_mot_color(idx=gallery_id+1), 2)
				draw_bb_text(frames[frame_idx], f'{class_name}, ID : {gallery_id}', (startX, startY, endX, endY), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, get_mot_color(idx=gallery_id+1))


def display_cross_person_reid(frames, list_of_video_fps):
	for frame, input_path, video_fps in zip(frames, video_filenames, list_of_video_fps):
		cv2.imshow(f'ReID output for {input_path}', frame)
		key = cv2.waitKey(int(1000//video_fps))


def close_windows(list_of_videos):
	# Release all videos
	for video in list_of_videos:
		video.release()

	# Closing windows
	for input_path in video_filenames:
		cv2.destroyWindow(f'ReID output for {input_path}')

### VISUALIZATION ROUTINE END ###




if __name__ == '__main__':

	list_of_videos, list_of_video_fps = get_list_of_videos()
	current_frame_index = STARTING_FRAME_IDX
	previous_feature_vectors_across_frames = []

	# Used for accuracy computation
	num_correct_detections, num_gt_total = [], []

	while(True):
		# start_time = time.time()
		frames, success = extract_next_frames(list_of_videos)
		if not success:
			print("A frame was not successfully extracted.")
			break
		
		# Skipping unnecessary frames
		if current_frame_index < STARTING_FRAME_IDX:
			current_frame_index += 1
			continue

		filtered_detections_across_frames, cropped_people_across_frames = perform_yolo_detections(frames)
		# Saving Yolo results
		# save_yolo_detections(filtered_detections_across_frames, current_frame_index)

		current_feature_vectors_across_frames = get_current_feature_vectors_across_frames(cropped_people_across_frames)
		
		if current_frame_index == STARTING_FRAME_IDX or not previous_feature_vectors_across_frames:
			previous_feature_vectors_across_frames = copy.deepcopy(current_feature_vectors_across_frames)
		else:
			# Compare current and previous feature vectors of people
			new_vs_new_feature_comparisons, new_vs_old_feature_comparisons = get_feature_comparisons(current_feature_vectors_across_frames, previous_feature_vectors_across_frames)
			
			# Get initial bboxes
			bboxes_across_frames = person_reid_through_time(new_vs_old_feature_comparisons,\
				  current_feature_vectors_across_frames,\
				  previous_feature_vectors_across_frames,\
				  filtered_detections_across_frames)
			
			# Refine initial bboxes (after this you can identify the same person across different frames)
			bboxes_across_frames = person_reid_through_frames(new_vs_new_feature_comparisons,\
			        current_feature_vectors_across_frames,\
					bboxes_across_frames)
			
			# Saving ReID results
			# save_reid_results(bboxes_across_frames, current_frame_index)
			
			# Draws bounding boxes on top of frames
			create_cv2_bboxes_across_frames(bboxes_across_frames, frames)
			
			# Perform evaluation
			_, num_correct_dets_across_frames, num_gt_total_across_frames = get_accuracy_score(current_frame_index + 1, bboxes_across_frames)
			num_correct_detections.append(num_correct_dets_across_frames)
			num_gt_total.append(num_gt_total_across_frames)
		
		# end_time = time.time()
		# print(f"Time to compute current set of frames: {end_time - start_time}")

		# Displaying the accuracy after 50 frames
		current_frame_index += 1
		print(f"Frame number is {current_frame_index}")
		if current_frame_index == STARTING_FRAME_IDX + 50:
			print(f"Number of correct detections: {sum(num_correct_detections)}")
			print(f"Number of total ground truths boxes: {sum(num_gt_total)}")
			print(f"Accuracy is: {sum(num_correct_detections) / sum(num_gt_total)}")

		# Showing the windows
		display_cross_person_reid(frames, list_of_video_fps)

	close_windows(list_of_videos)
