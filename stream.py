import time
import cv2
import argparse
import numpy as np
from imutils.video import VideoStream
import imutils
from yolo_utils import speak
# Cai dat tham so doc weight, config va class name
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', default='./yolov3-coco/tiny.cfg',
                help='path to yolo config file')
ap.add_argument('-w', '--weights', default='./yolov3-coco/tiny.weights',
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', default='./yolov3-coco/coco_vn-labels',
                help='path to text file containing class names')
args = ap.parse_args()

LIST_LABELS = {
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
last_object_name = ''
# Ham tra ve output layer
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# Ham ve cac hinh chu nhat va ten class
def draw_prediction(img, class_id, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



# Doc tu webcam
cap  = VideoStream(src=0).start()

# Doc ten cac class
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet(args.weights, args.config)


# Bat dau doc tu webcam
i=1
while (True):
    # Doc frame
    frame = cap.read()
    image = imutils.resize(frame, width=320)
    i+=1
    if i%10==0:
        # Resize va dua khung hinh vao mang predict
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        # Loc cac object trong khung hinh
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                #print(str(classes[class_id]))
                confidence = scores[class_id]
                #print('confidence:', confidence)
                if (confidence > 0.5):
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        print(class_ids)
        print(last_object_name)
        for id in class_ids:
            if id != 0:
                for label in LIST_LABELS:
                    if (classes[id]) == label and classes[id] != last_object_name:
                        last_object_name = classes[id]
                        speak(LIST_LABELS[label])
                        pass
        # Ve cac khung chu nhat quanh doi tuong
#         for i in indices:
#             box = boxes[i]
#             x = box[0]
#             y = box[1]
#             w = box[2]
#             h = box[3]
#             draw_prediction(image, class_ids[i], round(x), round(y), round(x + w), round(y + h))
        cv2.imshow("object detection", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
cv2.destroyAllWindows()