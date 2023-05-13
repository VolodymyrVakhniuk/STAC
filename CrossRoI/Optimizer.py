import gurobipy as gp
from gurobipy import GRB
import numpy as np
import CreateGraph

def optimization_solver(cameras, cam_to_tshape, time_window, gt_multi_hashmap=None):

    # Getting the number of tiles needed to cover a frame of particular camera.
    # Recall: Cameras may have different resolutions.
	cam_to_tcount = {cam: cam_to_tshape[cam][0]*cam_to_tshape[cam][1] for cam in cam_to_tshape}

    # Recall: multi_hashmap := (cam) -> (hashmap_tiles)
    # where hashmap_tiles := (frame, obj_id) -> (tiles)
	multi_hashmap, time_to_obj = CreateGraph.multi_cam_hashmap(cameras, time_window, gt_multi_hashmap)

	try:
		time_obj_len = sum([len(each) for each in time_to_obj.values()])

		# Create a new model
		model = gp.Model("optimization")

		x_shape_list = []
		for i in range(len(cameras)):
			x_shape_list.extend([(i, j) for j in range(cam_to_tcount[cameras[i]])])

		# Create variables
		x = model.addVars(x_shape_list, vtype=GRB.INTEGER, lb=[0]*len(x_shape_list), ub=[1]*len(x_shape_list))
		I = model.addVars(time_obj_len, len(cameras), vtype=GRB.BINARY)

		print("Variables added")

		# Set objective
		model.setObjective(gp.quicksum(x), GRB.MINIMIZE)

		# Add Constraints
		counter = 0
		for t in range(time_window[1]):
			for obj in time_to_obj[t]:
				for c in range(len(cameras)):
					if (t, obj) not in multi_hashmap[cameras[c]]:
						model.addConstr(I[counter, c] == 0)
					else:
						p = np.zeros(cam_to_tcount[cameras[c]])
						
						for pos in multi_hashmap[cameras[c]][(t, obj)]:
							p[pos] = 1
						model.addConstr((I[counter, c] == 1) >> \
							(gp.quicksum([p[j] - x[c, j] * p[j] for j in range(cam_to_tcount[cameras[c]])]) == 0))
						model.addConstr((I[counter, c] == 0) >> \
							(gp.quicksum([p[j] - x[c, j] * p[j] for j in range(cam_to_tcount[cameras[c]])]) >= 1))
				model.addConstr(gp.quicksum(I[counter, c] for c in range(len(cameras))) >= 1)
				counter += 1

		print("Constraints added")

		# Optimize model
		model.optimize()

	except gp.GurobiError as e:
		print('Error code ' + str(e.errno) + ": " + str(e))

	except AttributeError:
		print('Encountered an attribute error')

	# if x.getAttr('Status') != GRB.OPTIMAL:
	# 	print("WARNING: Was not able to find optimal solution")

	result = {cam: [] for cam in cameras}
	for i in range(len(cameras)):
		for j in range(cam_to_tcount[cameras[i]]):
			if round(x[i,j].x) == 1.0:
				result[cameras[i]].append(j)

	return result