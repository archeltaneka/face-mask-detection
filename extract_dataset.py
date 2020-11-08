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

print('Initiating dataset builder...')

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-a', '--annotations', required=True, help='Input path to your "annotations" directory (Download avaialable on: https://www.kaggle.com/andrewmvd/face-mask-detection')
arg_parser.add_argument('-i', '--images', required=True, help='Input path to your "images" directory (Download avaialable on: https://www.kaggle.com/andrewmvd/face-mask-detection')
arg_parser.add_argument('-s', '--size', required=True, help='Desired output image size')
arg_parser.add_argument('-o', '--output', required=True, help='Output path to the serialized (pickled) dataset')
args = vars(arg_parser.parse_args())

print('[INFO] Reading annotations...')
annotation_files = [f for f in listdir(args['annotations']) if isfile(join(args['annotations'], f))]

BASE = args['annotations']

names = []
bounding_boxes = []
img_paths = []
n_masks = []

# print(annotation_files[0])

for a in tqdm(annotation_files):
    a = os.path.join(BASE, a)
#     print(a)
#     break
    with open(a, 'rb') as f:
        content = f.read()
#     print(content)
    soup = BeautifulSoup(content, 'xml')

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

for i in tqdm(range(len(img_paths))):
    full_path = os.path.join(BASE, img_paths[i])
    img = cv2.imread(full_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    while(j < len(bounding_boxes)):
        for k in range(n_masks[i]):
            face = img[bounding_boxes[j][1]:bounding_boxes[j][3], bounding_boxes[j][0]:bounding_boxes[j][2]]
            faces.append(face)
        
            j += 1
            
        break

print('[STATUS] Task completed: read images & extract faces')
        
IMG_SIZE = args['size']
print('[INFO] Resizing faces according to the input size: {}x{}....'.format(IMG_SIZE, IMG_SIZE))

encoded_labels = []
unique_labels = []

for i in tqdm(range(len(faces))):
    faces[i] = cv2.resize(faces[i], (int(IMG_SIZE), int(IMG_SIZE)))

print('[STATUS] Task completed: Resize faces')
print('[INFO] Building data labels...')
    
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
print('Building dataset completed!')