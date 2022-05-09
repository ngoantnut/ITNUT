import speech_recognition as sr
import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
import IPython.display as ipd
from pydub import AudioSegment
import os
import vlc 
import os
import datetime
import playsound
from gtts import gTTS

CONFIDENCE = 0.5
THREHOLD = 0.3


def show_image(img):
    cv.imshow("Image", img)
    cv.waitKey(0)


def draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels):
    # If there are any detections
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Get the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            # Get the unique color for this class
            color = [int(c) for c in colors[classids[i]]]
            print(color)
            # Draw the bounding box rectangle and label on the image
            cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
            text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
            cv.putText(img, text, (x, y-5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img, labels[classids[i]]


def generate_boxes_confidences_classids(outs, height, width, tconf):
    boxes = []
    confidences = []
    classids = []

    for out in outs:
        for detection in out:
            #print (detection)
            #a = input('GO!')

            # Get the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]

            # Consider only the predictions that are above a certain confidence level
            if confidence > tconf:
                # TODO Check detection
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')

                # Using the center x, y coordinates to derive the top
                # and the left corner of the bounding box
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))

                # Append to list
                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                classids.append(classid)

    return boxes, confidences, classids


def TextPreProcessing(text):
    words_list = text.split(" ")
    text_to_words = []
    for word in words_list:
        text_to_words.append(word.lower())
    return text_to_words


def TextToSpeech(text):
    text_array = TextPreProcessing(text)
    AUDIO_DIR = "/home/chessie/Documents/Testing-Dir/YOLOv3-Object-Detection-with-OpenCV/banmai /"
    mp3_path_array = []
    for word in text_array:
        mp3_path_array.append(AUDIO_DIR+word+".mp3")
    result = 0
    # for path in mp3_path_array:
    #     result += AudioSegment.from_mp3(path)
    # result.export("KQ.mp3", format="mp3")
    # ipd.Audio('KQ.mp3')
    # file = "/home/chessie/Documents/Testing-Dir/RasPI-Prj/KQ.mp3"
    # print(mp3_path_array[0])
    # audio = vlc.MediaPlayer(mp3_path_array[0])
    # audio.play()
    playsound.playsound(mp3_path_array[0], True)

def infer_image(net, layer_names, height, width, img, colors, labels, FLAGS,
                boxes=None, confidences=None, classids=None, idxs=None, infer=True):

    if infer:
        # Contructing a blob from the input image
        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                                    swapRB=True, crop=False)

        # Perform a forward pass of the YOLO object detector
        net.setInput(blob)

        # Getting the outputs from the output layers
        start = time.time()
        outs = net.forward(layer_names)
        end = time.time()

        # if FLAGS.show_time:
        #     print("[INFO] YOLOv3 took {:6f} seconds".format(end - start))

        # Generate the boxes, confidences, and classIDs
        boxes, confidences, classids = generate_boxes_confidences_classids(
            outs, height, width, CONFIDENCE)

        # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
        idxs = cv.dnn.NMSBoxes(
            boxes, confidences, CONFIDENCE, THREHOLD)

    if boxes is None or confidences is None or idxs is None or classids is None:
        raise '[ERROR] Required variables are set to None before drawing boxes on images.'

    # Draw labels and boxes on the image
    img, name = draw_labels_and_boxes(
        img, boxes, confidences, classids, idxs, colors, labels)
    print(name)
    return img, boxes, confidences, classids, idxs, name


def speak(text):
    print("Bot: {}".format(text))
    tts = gTTS(text=text, lang= "vi", slow=False)
    tts.save("sound.mp3")
    playsound.playsound("sound.mp3", False)
    os.remove("sound.mp3")

def  speech_recoginiton():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Nói gì đó nào bé ơiiii!")
        audio = r.listen(source)

    # Speech recognition using Google Speech Recognition
    try:
        print("Bạn đã nói: " + r.recognize_google(audio, language="vi-VN"))
    except sr.UnknownValueError:
        print("Google Speech Recognition không thể nhận dạng tiếng nói")
    except sr.RequestError as e:
        print("Không thể kết nối tới máy chủ; {0}".format(e))
