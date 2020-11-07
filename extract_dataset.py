from glob import glob
import numpy as np
import argparse
import pickle
import cv2
import os

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-a', '--annotations', required=True, help='Input path to your "annotations" directory (Download avaialable on: https://www.kaggle.com/andrewmvd/face-mask-detection')
arg_parser.add_argument('-i', '--images', required=True, help='Input path to your "images" directory (Download avaialable on: https://www.kaggle.com/andrewmvd/face-mask-detection')
arg_parser.add_argument('-o', '--output', required=True, help='Output path to the serialized (pickled) dataset')
args = vars(arg_parser.parse_args())

print('[INFO] Reading annotations...')
annotation_files = glob(args['annotations'])

names = []
bounding_boxes = []
img_paths = []
n_masks = []

