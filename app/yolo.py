import cv2 
import streamlit as st
import numpy as np 
import imutils 
import time 
import os 
import config 

# loading image 

def object_detection(video, confidence_threshold, nms_threshold): 
    net = cv2.dnn.readNet(config.CONFIG_PATH, config.MODEL_PATH) 
    layer_names = net.getLayerNames() 
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()] 

    vs = cv2.VideoCapture(video) 
    fps = vs.get(cv2.CAP_PROP_FPS) 
    writer_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    writer_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None 
    (W, H) = (None, None) 

    try:
        prop = cv2.CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop)) 
        print(f"[INFO] {total} frames in the video") 
         
    except:
        print(f"[INFO] {total} frames in the video") 
        total = -1

    
    while True:
        (grabbed, frame) = vs.read()

		# if no frame is grabbed, we reached the end of video, so break the loop 
        if not grabbed:
            break
		# if the frame dimensions are empty, grab them 
        if W is None or H is None:
            (H, W) = frame.shape[:2]

		# build blob and feed forward to YOLO to get bounding boxes and probability 
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB = True, crop=False)
        start = time.time() 
        net.setInput(blob) 
        layerOutputs = net.forward(output_layers) 
        end = time.time() 

    # output the bounding box and class 
        classIDs = []
        confidences = []
        boxes = []
        for output in layerOutputs: 
            for detection in output: 
                score = detection[5:]
                classID = np.argmax(score)
                confidence = score[classID]
                if confidence > confidence_threshold: 
                    box = detection[0:4] * np.array([W, H, W, H]) 
                    (centerX, centerY, width, height) = box.astype('int')

                    # Rectangle coordinates
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold) 
        if len(idxs) > 0: 
            for i in idxs.flatten():
                (x,y) = (boxes[i][0], boxes[i][1]) 
                (w,h) = (boxes[i][2], boxes[i][3]) 
                color = [int(c) for c in config.COLORS[classIDs[i]]] 
                cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2) 
                text = f"{config.LABELS[classIDs[i]]}: {confidences[i]}" 
                cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'MPEG') 
            writer = cv2.VideoWriter(config.OUTPUT_PATH, fourcc, fps, (writer_width, writer_height), True)

            if total > 0:
                elap = (end - start) 
                print(f"[INFO] single frame took {round(elap/60,2)} minutes") 
                print(f"[INFO] total estimated time to finish: {(elap*total)/60} minutes") 

        writer.write(frame) 

    writer.release() 
    vs.release() 

    return total, elap
