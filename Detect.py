import argparse
import sys
import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import numpy as np
import math
from trackeable import TrackableObject
from datetime import datetime
from upload import GoogleSheet

counter, fps = 0, 0
fps_avg_frame_count = 10
start_time = time.time()

cap = cv2.VideoCapture(0)
width = 640#int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = 380#int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

base_options = core.BaseOptions(file_name = 'models/Dataset2000/custommodel00.tflite')#'models/custommodel.tflite')
detection_options = processor.DetectionOptions(max_results = 100, score_threshold = 0.35) ### cambiar
options = vision.ObjectDetectorOptions(base_options = base_options, detection_options = detection_options)
detector = vision.ObjectDetector.create_from_options(options)

centerPointsPrevFrame = []
trackingObjects = {}
trackId = 0

roi_position_entry = 0.50 #rigth
roi_position_exit = 0.50 #left

position = [0,0,0,0] #left, right, up, down;
trackableobject = {}
Eje = True # x = True, y = False

sheet = GoogleSheet()
sheet.lengthLeftRigth()
while True:
    if trackId > 99:
        trackId = 0
        trackableobject = {}
        
    _, frame = cap.read()
    
    counter +=1
    objects = []
    centerPointsCurFrame = []
    
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (width,height))  
    
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #print(rgb_image)
    
    input_tensor = vision.TensorImage.create_from_array(rgb_image)
    
    detection_result = detector.detect(input_tensor)
    
    
    for detection in detection_result.detections:
        
        category = detection.categories[0]
        category_name = category.category_name
        if category_name == 'person':
        
            probability = round(category.score, 2)        
            bbox = detection.bounding_box
            #print(bbox)
            star_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(frame, star_point, end_point, (0,255,0), 2)
            
            result_text = category_name + '('+str(probability)+')'
            text_location = (bbox.origin_x + 10, -10 + bbox.origin_y)
            
            cv2.putText(frame, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
            
            xmin = star_point[0]
            xmax = end_point[0]
            ymin = star_point[1]
            ymax = end_point[1]
            xcenter = xmin + (int(round((xmax - xmin )/ 2)))
            ycenter = ymin + (int(round((ymax - ymin )/ 2)))
            #cv2.circle(frame, (xcenter, ycenter), 5, (0,255,0), -1)
            centerPointsCurFrame.append((xcenter, ycenter))
            objects.append((xmin,ymin,xmax,ymax))
            
    if counter <= 2:        
        for pt in centerPointsCurFrame:
            for pt2 in centerPointsPrevFrame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1]-pt[1])
                
                if distance < 10:
                    trackingObjects[trackId] = pt
                    trackId += 1
    else:
        
        trackingObjects_copy = trackingObjects.copy()
        centerPointsCurFrame_copy = centerPointsCurFrame.copy()
        for objectId, pt2 in trackingObjects_copy.items():
            object_exists = False
            for pt in centerPointsCurFrame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1]-pt[1])
                if distance < 20:
                    trackingObjects[objectId] = pt
                    object_exists = True
                    if pt in centerPointsCurFrame:
                        centerPointsCurFrame.remove(pt)
                    continue
                
            if not object_exists:
                trackingObjects.pop(objectId)
    
    for pt in centerPointsCurFrame:
        trackingObjects[trackId] = pt
        trackId +=1
    
    counted = False
    
    for objectId, pt in trackingObjects.items():
        to = trackableobject.get(objectId, None)
        
        if to is None:
            to = TrackableObject(objectId, pt)
        else:
            if Eje and not to.counted:
                Xpos = [c[0] for c in to.centroids]
                direction = pt[0] - np.mean(Xpos)
                
                if pt[0] > roi_position_entry*width and direction > 0 and np.mean(Xpos) < roi_position_entry*width:
                    position[1] += 1
                    to.counted = True
                    now = datetime.now().strftime('%H:%M:%S')
                    sheet.sendData('Entry', now)
                    sheet.lenRight += 1
                    
                elif pt[0] < roi_position_exit*width and direction < 0 and np.mean(Xpos) > roi_position_exit*width:    
                #elif pt[0] < roi_position_entry*width and direction < 0 and np.mean(Xpos) > roi_position_entry*width:
                    position[0] += 1
                    to.counted = True
                    now = datetime.now().strftime('%H:%M:%S')
                    sheet.sendData('Exit', now)
                    sheet.lenLeft += 1
                    
            to.centroids.append(pt)
        trackableobject[objectId] = to
        
        cv2.circle(frame, pt, 5, (0,255,0), -1)
        cv2.putText(frame, str(objectId), (pt[0], pt[1] - 7),0, 1, (0, 0, 255), 2)
    
    if counter % fps_avg_frame_count == 0:
        end_time = time.time()
        fps = fps_avg_frame_count / (end_time - start_time)
        start_time = time.time()
        
    fps_text = 'FPS = {:.1f}'.format(fps)
    cv2.putText(frame, fps_text, (24,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.line(frame, (int(roi_position_entry*width), 0),(int(roi_position_entry*width), height), (255, 0, 0), 5)
    cv2.line(frame, (int(roi_position_exit*width), 0),(int(roi_position_exit*width), height), (0, 0, 255), 5)
    cv2.putText(frame, f'Entrada:{sheet.lenRight}; Salida: {sheet.lenLeft}',(10,35), 2,1, (0, 0, 0), 2, cv2.FONT_HERSHEY_SIMPLEX )

    cv2.imshow('frame', frame)
    #cv2.imshow('rgb', rgb_image)
    
    centerPointsPrevFrame = centerPointsCurFrame.copy()
    
    if cv2.waitKey(1) == 27:
        break
        
cap.release()
cv2.destroyAllWindows()
    