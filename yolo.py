import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os

from pip import main
from yolo_utils import *

FLAGS = []
MODEL_PATH = './yolov3-coco/'
WEIGHT_PATH = './yolov3-coco/yolov3.weights'
CONFIG_PATH = './yolov3-coco/yolov3.cfg'
LABEL_PATH = './yolov3-coco/coco_vn-labels'
CONFIDENCE = 0.5
THREEHOLD = 0.3


# Get the labels
labels = open(LABEL_PATH).read().strip().split('\n')
# Intializing colors to represent each label uniquely
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# Load the weights and configutation to form the pretrained YOLOv3 model
net = cv.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHT_PATH)

# Get the output layer names of the model
layer_names = net.getLayerNames()
layer_names = [layer_names[i - 1]
               for i in net.getUnconnectedOutLayers()]
vid = cv.VideoCapture(0)
last_object_name = ''
list_labels = {
    # 'person': 'Đây là người bạn nhé',
    'bicycle': 'Đây là cái xe đạp',
    'car': 'Đây là ô tô nha',
    'motorbike': 'Đây là xe máy nhé',
    'aeroplane': 'Đây là máy bay nhé',
    'bus': 'Đây là xe buýt nhé',
    'train': 'Đây là tàu hỏa',
    'truck': 'Đây là xe tải',
    'boat': 'Đây là thuyền',
    'traffic light': 'Đây là đèn giao thông',
    'fire hydrant': 'Đây là trụ nước chữa cháy',
    'stop sign': 'Đây là cảnh báo dừng lại',
    'parking meter': 'nơi đỗ xe',
    'bench': 'đây là ghế dài',
    'bird': 'đây là chim',
    'cat': 'đây là mèo',
    'dog': 'đây là chó',
    'horse': 'đây là ngựa',
    'sheep': 'đây là cừu',
    'cow': 'đây là bò',
    'elephant': 'đây là voi',
    'bear': 'đây là gấu',
    'zebra': 'đây là ngựa vằn',
    'giraffe': 'đây là hươu cao cổ',
    'backpack': 'đây là cặp sách',
    'umbrella': 'đây là cái ô',
    'handbag': 'đây là túi xách',
    'tie': 'đây là cà vạt',
    'suitcase': 'đây là va li',
    'frisbee': 'đây là đĩa ném',
    'skis': 'đây là đĩa ván trượt',
    'snowboard': 'đây là ván trượt tuyết',
    'sports ball': 'đây là bóng đá',
    'kite': 'đây là cái diều',
    'baseball bat': 'đây là mũ bóng chày',
    'baseball glove': 'đây là găng tay chơi bóng chày',
    'skateboard': 'đây là ván trượt',
    'surfboard': 'đây là ván lướt sóng',
    'tennis racket': 'đây là vợt tennis',
    'bottle': 'đây là cái bình',
    'wine glass': 'đây là ly rượu',
    'cup': 'đây là cái cốc',
    'fork': 'đây là cái dĩa',
    'knife': 'đây là con dao',
    'spoon': 'đây là cái thìa',
    'bowl': 'đây là cái bát',
    'banana': 'đây là qủa chuối',
    'apple': 'đây là quả táo',
    'sandwich': 'đây là bánh mỳ kẹp',
    'orange': 'đây là quả cam',
    'broccoli': 'đây là bông súp lơ',
    'carrot': 'đây là cà rốt',
    'hot dog': 'đây là xúc xích',
    'pizza': 'đây là bánh pi da',
    'donut': 'đây là bánh vòng ',
    'cake': 'đây là cái bánh',
    'chair': 'đây là cái ghế',
    'sofa': 'đây là ghé sô pha',
    'pottedplant': 'đây là cây',
    'bed': 'đây là cái giường',
    'diningtable': 'đây là bàn ăn',
    'toilet': 'đây là toi lét',
    'tvmonitor': 'đây là Ti vi',
    'laptop': 'đây là máy tính xách tay',
    'mouse': 'đây là con chuột',
    'remote': 'đây là cái điều khiển',
    'keyboard': 'đây là bàn phím',
    'cell phone': 'đây là điện thoại',
    'microwave': 'đây là lò vi sóng',
    'oven': 'đây là lò nướng',
    'toaster': 'đây là lò nướng bánh mỳ',
    'sink': 'đây là bồn rửa chén',
    'refrigerator': 'đây là tủ lạnh',
    'book': 'đây là quyển sách',
    'clock': 'đây là đồng hồ',
    'vase': 'đây là bình hoa',
    'scissors': 'đây là cái kéo',
    'teddy bear': 'đây là gấu tét đi',
    'hair drier': 'đây là mấy sấy tóc',
    'toothbrush': 'đây là bàn chải đánh răng',
}


# def recognize():
#     speak('Xin chào')
#     count = 0
#     _, frame = vid.read()
#     labels_of_img = ''
#     height, width = frame.shape[:2]
#     if count == 0:
#         frame, boxes, confidences, classids, idxs, labels_of_img = infer_image(net, layer_names,
#                                                                             height, width, frame, colors, labels, FLAGS)
#         count += 1
#     else:
#         frame, boxes, confidences, classids, idxs, labels_of_img = infer_image(net, layer_names,
#                                                                             height, width, frame, colors, labels, FLAGS, boxes, confidences, classids, idxs, infer=False)
#         count = (count + 1) % 6
#     # cv.imshow('webcam', frame)
#     for lable in list_labels:
#         if labels_of_img == lable:
#             print('check', lable)
#             print(list_labels[lable])
#             speak(list_labels[lable])
#             time.sleep(10)
#             vid.release()
#             cv.destroyAllWindows()
#             time.sleep(5)
#             recognize()


# if __name__ == '__main__':
#     while True:
#         recognize()


while True:
    count = 0
    _, frame = vid.read()
    labels_of_img = ''
    height, width = frame.shape[:2]
    if count == 0:
        frame, boxes, confidences, classids, idxs, labels_of_img = infer_image(net, layer_names,
                                                                               height, width, frame, colors, labels, FLAGS)
        count += 1
    else:
        frame, boxes, confidences, classids, idxs, labels_of_img = infer_image(net, layer_names,
                                                                               height, width, frame, colors, labels, FLAGS, boxes, confidences, classids, idxs, infer=False)
        count = (count + 1) % 6
    cv.imshow('webcam', frame)
    print('last_object_name check 1', last_object_name)
    for lable in list_labels:
        if labels_of_img == lable and labels_of_img != last_object_name:
            last_object_name = labels_of_img
            print('last_object_name check 2', last_object_name)
            speak(list_labels[lable])
            time.sleep(0.1)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv.destroyAllWindows()


