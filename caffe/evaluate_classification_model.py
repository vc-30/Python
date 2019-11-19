import sys
import os
import glob
import cv2
import numpy as np
import caffe

from caffe.proto import caffe_pb2
from argparse import ArgumentParser

caffe.set_mode_gpu()

label_map_file = "/path/to/your/label_map.txt"

def read_image_paths(img_file_list):
	valid_file_list = []
	with open(label_map_file) as f:
		labels = [l.strip().split(' ')[0] for l in f.readlines()]

	with open(img_file_list) as f:
		files = f.readlines()
		for file in files:
			gt_label = file.split("/")[-2]	#Assuming generate_classification_data.py was used to generate the data
			# print gt_label[-2]
			if gt_label in labels:
				valid_file_list.append(file.strip())
	return valid_file_list

def construct_net(deploy_file, weights_file):
	net = caffe.Net(deploy_file, weights_file, caffe.TEST)

	transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,1,0))

	return net, transformer

def Visualize(img, prob):
	with open(label_map_file) as f:
		labels = [l.strip().split(' ')[0] for l in f.readlines()]
	# print labels

	userfriendly_label = labels[prob.argmax()]

	# cv2.putText()
	just_to_display = cv2.resize(img, (200,200), interpolation = cv2.INTER_CUBIC)

	cv2.namedWindow(userfriendly_label, cv2.WINDOW_NORMAL)
	cv2.imshow(userfriendly_label, just_to_display)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def predict_output(net, transformer, img_list, visualization=False):
	# gt = []
	pred = []
	for img_path in img_list:
		#get input dim from network to resize input image
		_, _, img_width, img_height = net.blobs['data'].data.shape 
		img = cv2.imread(img_path, cv2.IMREAD_COLOR)
		img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

		#load image in blob
		net.blobs['data'].data[...] = transformer.preprocess('data', img)

		#Forward pass
		out = net.forward()

		#read predictions (probability) from output [softmax] layer
		prob = out['softmax']

		pred.append(prob.argmax()+1) #To convert from 0th index to 1st index
		if visualization:
			Visualize(img, prob)

	# print (prob.argmax())
	return pred

	# cv2.imshow("input", img)
	# cv2.waitKey(0)

def get_gt_labels(img_list):
	gts = []
	with open(label_map_file) as f:
		labels = [l.strip().split(' ')[0] for l in f.readlines()]

	# print img_list
	for img_path in img_list:
		# print img_path
		gt = img_path.split('/')[-2]
		gt_numeric_label = labels.index(gt) + 1 #Note assumption is networks output will always be in range of label_map
		gts.append(gt_numeric_label)

	return gts

if __name__ == '__main__':
	parser = ArgumentParser()

	parser.add_argument("-d", "--deploy", dest="deploy_file",
						help="deploy.prototxt expected", required=True)
	parser.add_argument("-m", "--model", dest="weights_file",
						help="caffemodel file expected", required=True)
	parser.add_argument("-il", "--img_list", dest="img_file_list",
						help="image file list expected", required=True)
	parser.add_argument("-v", "--visualization", dest="visualization", default=False,
						help="Enable this to also visualize the output. Note: you will have to modify \
						script to provide userfriendly label map file")
	parser.add_argument("-o", "--out_file_name", dest="outfile", required=True,
						help="Provide output filename (abs path) where output  needs to be stored")

	args = parser.parse_args()

	img_list = read_image_paths(args.img_file_list)

	net, transformer = construct_net(args.deploy_file, args.weights_file)

	predictions = predict_output(net, transformer, img_list, False)

	gts = get_gt_labels(img_list)

	# if not os.path.exists(args.outfile):
	# 	print "%s file doesn't exists"%args.outfile
	# 	exit(1)

	with open(args.outfile, "w") as f:
		for idx,gt in enumerate(gts):
			f.write(" ".join([str(gt), str(predictions[idx]), "\n"]))

