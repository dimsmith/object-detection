import os

import cv2
import numpy as np
from playsound import playsound

"""
    Global variables
"""
# Threshold to detect object
threshold = 0.45
nms_threshold = 0.2
CWD_PATH = os.getcwd()
classNames = []
classFile = os.path.join(CWD_PATH, "model", "coco.names")

if __name__ == "__main__":
    # Load model files
    with open(classFile, 'rt') as file:
        classNames = file.read().rstrip('\n').split('\n')
    configPath = os.path.join(CWD_PATH, "model", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    weightsPath = os.path.join(CWD_PATH, "model", "frozen_inference_graph.pb")

    # Init Video Capture
    cap = cv2.VideoCapture(0)

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    confidence_level_target = 70
    confidence_level = 0
    while True:
        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=threshold)
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1, -1)[0])
        confs = list(map(float, confs))

        indices = cv2.dnn.NMSBoxes(bbox, confs, threshold, nms_threshold)

        for i in indices:
            i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x, y), (x + w, h + y), color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classIds[i][0] - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            confidence_level = round(confs[0] * 100, 2)
            cv2.putText(img, str(confidence_level), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Output", img)
        # if confidence_level > confidence_level_target:
        #     playsound("beep.wav")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
