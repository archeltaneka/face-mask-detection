from lib.modelbuilder import Model

import numpy as np
import argparse
import torch
import cv2

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-d', '--detector', required=True, help='Path to face detector file')
arg_parser.add_argument('-r', '--recognition_model', required=True, help='Path to face recognition file')
arg_parser.add_argument('-m', '--model', required=True, help='Path to the saved model')
args = vars(arg_parser.parse_args())

IMG_SIZE = 64
LABELS = ['without_mask', 'with_mask', 'mask_weared_incorrect']
frame_name = 'Webcam Capture'

model_path = args['model']
prototxt_file = args['detector']
caffemodel_file = args['recognition_model']

use_cuda = torch.cuda.is_available
model = Model()
if use_cuda:
    model = model.cuda()
model.load_state_dict(torch.load(model_path))

cap = cv2.VideoCapture(0)
face_model = cv2.dnn.readNetFromCaffe(prototxt_file, caffemodel_file)

if cap is None or not cap.isOpened():
    print('[ERROR] No camera device is detected!')
else:
    while True:
        ret, frame = cap.read()
        (h,w) = frame.shape[:2]
        face_blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300,300), mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
        face_model.setInput(face_blob)
        detector = face_model.forward()

        if len(detector) > 0:
            for i in range(0, detector.shape[2]):
                rect = detector[0,0,i,3:7] * np.array([w,h,w,h])
                (start_x, start_y, end_x, end_y) = rect.astype('int')
                confidence = detector[0,0,i,2]

                if confidence > 0.5:
                    face = frame[start_y:end_y, start_x:end_x]
                    if face.size == 0:
                        continue

                    resized_face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                    resized_face = np.expand_dims(resized_face, axis=0)
                    resized_face = np.reshape(resized_face, [1,3,IMG_SIZE,IMG_SIZE])

                    test_feature = torch.Tensor(resized_face)
                    model.eval()
                    if use_cuda:
                        test_feature = test_feature.cuda()

                    output = model(test_feature)
                    pred = np.argmax(output.cpu().detach())
                    e_x = np.exp(output[0].cpu().detach().numpy())
                    score = np.max(e_x / e_x.sum() * 100)

                    text = '{}: {:.2f}%'.format(LABELS[pred], score)

                    if pred == 0:
                        cv2.putText(frame, text, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
                        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255,0,255), 2)
                    elif pred == 1:
                        cv2.putText(frame, text, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,0), 2)
                        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0,128,0), 2)
                    else:
                        cv2.putText(frame, text, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
                        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255,255,0), 2)

        cv2.imshow(frame_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
