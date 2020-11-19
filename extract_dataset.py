from glob import glob
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from bs4 import BeautifulSoup

import numpy as np
import argparse
import pickle
import cv2
import os

"""
Extract and prepare dataset
"""

print('Initiating dataset builder...')

# adds argument parser for the script
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-a', '--annotations', required=True, help='Input path to your "annotations" directory (Download avaialable on: https://www.kaggle.com/andrewmvd/face-mask-detection') # path to your 'annotations' directory
arg_parser.add_argument('-i', '--images', required=True, help='Input path to your "images" directory (Download avaialable on: https://www.kaggle.com/andrewmvd/face-mask-detection') # path to your 'images' directory
arg_parser.add_argument('-o', '--output', required=True, help='Output path to the serialized (pickled) dataset') # specify the output path of the extracted dataset
args = vars(arg_parser.parse_args())

print('[INFO] Reading annotations...')
annotation_files = [f for f in listdir(args['annotations']) if isfile(join(args['annotations'], f))] # list all files inside the 'annotations' directory

BASE = args['annotations']

names = []
bounding_boxes = []
img_paths = []
n_masks = []

# read in all annotation files
for a in tqdm(annotation_files):
    a = os.path.join(BASE, a)
    with open(a, 'rb') as f:
        content = f.read()

    # read the contents of .xml files
    soup = BeautifulSoup(content, 'xml')

    # extract information/attribute: filename, xmin, xmax, ymin, ymax, names, bounding boxes, and number of faces
    filename = soup.find('filename').text
    objects = soup.find_all('object')
    n = 0
    for o in objects:
        name = o.find('name').text
        xmin = int(o.find('xmin').text)
        xmax = int(o.find('xmax').text)
        ymin = int(o.find('ymin').text)
        ymax = int(o.find('ymax').text)

        names.append(name)
        bounding_boxes.append([xmin, ymin, xmax, ymax])

        n = n + 1

    img_paths.append(filename)
    n_masks.append(n)

print('[STATUS] Task completed: Reading annotations')
print('[INFO] Reading images & extracting faces...')

BASE = args['images']
faces = []
j = 0

# read in image files from the extracted 'filename' attribute
for i in tqdm(range(len(img_paths))):
    full_path = os.path.join(BASE, img_paths[i])
    img = cv2.imread(full_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # switch BGR to RGB color channel
    
    # crop faces according to the corresponding bounding boxes
    while(j < len(bounding_boxes)):
        for k in range(n_masks[i]):
            face = img[bounding_boxes[j][1]:bounding_boxes[j][3], bounding_boxes[j][0]:bounding_boxes[j][2]]
            faces.append(face)
            
            j += 1
            
        break

print('[STATUS] Task completed: Read images & extract faces')

# WARNING: you can change this accordingly. If you want to make changes, make sure you also set the same value in 'train_and_evaluate_model.py' and 'open_and_predict_from_webcam.py' files
IMG_SIZE = 64 # image resizing value
print('[INFO] Resizing faces according to the input size: {}x{}....'.format(IMG_SIZE, IMG_SIZE))

encoded_labels = []
unique_labels = []

# resize faces to IMG_SIZE size
for i in tqdm(range(len(faces))):
    faces[i] = cv2.resize(faces[i], (int(IMG_SIZE), int(IMG_SIZE)))

print('[STATUS] Task completed: Resize faces')
print('[INFO] Building data labels...')

# convert 'names' attribute to numerical labels
for l in tqdm(names):
    if l not in unique_labels:
            unique_labels.append(l)
    if l == 'without_mask':
        encoded_labels.append(0)
    elif l == 'with_mask':
        encoded_labels.append(1)
    else:
        encoded_labels.append(2)
        
print('[STATUS] Task completed: Building data labels')
print('[INFO] Saving data into an output file...')
print('[STATUS] Saving completed:', args['output'])

# dict object to save it into a file
data_pickle = {'features': faces, 'labels': encoded_labels}

# save into a pickle file
with open(args['output'], 'wb') as f:
    pickle.dump(data_pickle, f)
    
print('Building dataset completed!\n')

# dataset statistics
print("Total number of images:", len(faces))
print("Total number of unique labels:", len(unique_labels))
print("Labels:", unique_labels)
print("'with_mask' labeled images:", sum(n == 'with_mask' for n in names))
print("'without_mask' labeled images:", sum(n == 'without_mask' for n in names))
print("'mask_weared_incorrect' labeled images:", sum(n == 'mask_weared_incorrect' for n in names))