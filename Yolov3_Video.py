import cv2
import numpy as np
import imutils
import time

net=cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
classes=[]
with open('coco.names','r') as f:
    classes=[line.strip() for line in f.readlines()]
#print(classes)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

video=cv2.VideoCapture("Path_to_the_video")
writer = None
(W, H) = (None, None)

try:
    if imutils.is_cv2():
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT 
    else:
        prop = cv2.CAP_PROP_FRAME_COUNT
    total = int(video.get(prop))
    print("[INFO] {} total frames in video".format(total))
    
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

while True:
    (grabbed, frame) = video.read()
    
    if not grabbed:
        break
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(output_layers)
    end = time.time()
    
    boxes = []
    confidences = []
    classIDs = []
    
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]],confidences[i])
            cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        writer = cv2.VideoWriter("NewVideo.mp4", fourcc, 30,(frame.shape[1], frame.shape[0]), True)
        
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))
            
    writer.write(frame)

print("[INFO] cleaning up...")
writer.release()
video.release()
