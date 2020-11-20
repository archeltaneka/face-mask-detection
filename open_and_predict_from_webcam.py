from lib.modelbuilder import Model

import numpy as np
import argparse
import torch
import cv2

print('[INFO] Initiating...')
print('[INFO] Loading resources...')

# adds argument parser for the script
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-d', '--detector', required=True, help='Path to face detector file') # path to your 'deploy.prototxt.txt' file
arg_parser.add_argument('-r', '--recognition_model', required=True, help='Path to face recognition file') # path to your 'caffemodel.weights' file
arg_parser.add_argument('-m', '--model', required=True, help='Path to the saved model') # path to your saved trained model
args = vars(arg_parser.parse_args())

print('[INFO] Model used for prediction:', args['model'])
print('[STATUS] Loading resources completed')

# WARNING: you can change this accordingly. If you want to make changes, make sure you also set the same value in 'extract_dataset.py' and 'train_and_evaluate.py' files
IMG_SIZE = 64 # resize value
LABELS = ['without_mask', 'with_mask', 'mask_weared_incorrect'] # dataset labels
frame_name = 'Webcam Capture'

# save argument parser into variables
model_path = args['model']
prototxt_file = args['detector']
caffemodel_file = args['recognition_model']

# check for CUDA and GPU computation availability
use_cuda = torch.cuda.is_available
# initiate model object
model = Model()
if use_cuda:
    model = model.cuda() # move model to GPU
# load the saved model from the training phase
model.load_state_dict(torch.load(model_path))

# prepare video capture with webcam/camera
cap = cv2.VideoCapture(0)
# load in the face detection model
face_model = cv2.dnn.readNetFromCaffe(prototxt_file, caffemodel_file)

print('[INFO] Looking for camera/webcam...')

# check for webcam/camera availability
if cap is None or not cap.isOpened():
    print('[ERROR] No camera device is detected!')
else:
    print('[INFO] Camera found! Opening...')
    while True:
        # read webcam/camera frames
        ret, frame = cap.read()
        (h,w) = frame.shape[:2]
        # detect any faces found
        face_blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300,300), mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
        face_model.setInput(face_blob)
        detector = face_model.forward()

        if len(detector) > 0:
            for i in range(0, detector.shape[2]):
                rect = detector[0,0,i,3:7] * np.array([w,h,w,h]) # for any faces found, prepare a rectangle bounding box
                (start_x, start_y, end_x, end_y) = rect.astype('int') # bounding box position (x and y)
                confidence = detector[0,0,i,2] # prediction confidence

                # proceed to the next step if confidence is above the threshold
                if confidence > 0.5:
                    face = frame[start_y:end_y, start_x:end_x]
                    if face.size == 0:
                        continue
                    
                    resized_face = cv2.resize(face, (IMG_SIZE, IMG_SIZE)) # resize face
                    resized_face = np.expand_dims(resized_face, axis=0) # expand dimension from 3D (IMG_SIZE, IMG_SIZE, 3) to 4D (IMG_SIZE, IMG_SIZE, m, c)
                    resized_face = np.reshape(resized_face, [1,3,IMG_SIZE,IMG_SIZE]) # reshape dimension to (m, c, IMG_SIZE, IMG_SIZE)

                    test_feature = torch.Tensor(resized_face) # convert numpy.array to torch.Tensor
                    model.eval() # set model to evaluation mode
                    if use_cuda:
                        test_feature = test_feature.cuda() # move to GPU if available
                    
                    # predict using the trained model
                    output = model(test_feature)
                    pred = np.argmax(output.cpu().detach())
                    # calculate prediction score using Softmax formula
                    e_x = np.exp(output[0].cpu().detach().numpy()) # e^x
                    score = np.max(e_x / e_x.sum() * 100) # max(e^x / sum(e^x))

                    # set prediction text
                    text = '{}: {:.2f}%'.format(LABELS[pred], score)
                    if pred == 0: # without mask
                        cv2.putText(frame, text, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
                        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255,0,255), 2)
                    elif pred == 1: # with mask
                        cv2.putText(frame, text, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,0), 2)
                        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0,128,0), 2)
                    else: # mask weared incorrect
                        cv2.putText(frame, text, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
                        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255,255,0), 2)

        cv2.imshow(frame_name, frame)

        # terminate program on 'q' keyboard press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print('[INFO] Closing program...')
    cap.release()
    cv2.destroyAllWindows()
