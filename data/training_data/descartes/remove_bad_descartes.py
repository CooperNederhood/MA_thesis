'''
There are two bad images of settlements that we should clean from the training data
They are:
	ona_id28_image.png / ona_id67_image.png
'''

import os 

path = "RGB/segmentation/size_256"
to_delete = ["ona_id28_image", "ona_id67_image"]

to_delete_files = []
for t in ["train", "val"]:
	for s in ["mask", "image"]:

		all_files = os.listdir(os.path.join(path, t, s))

		for f in all_files:
			for bad_file in to_delete:
				if bad_file in f:
					to_delete_files.append(f)
					print("Will delete: ", os.path.join(path, t, s, f))
					os.remove(os.path.join(path, t, s, f))



