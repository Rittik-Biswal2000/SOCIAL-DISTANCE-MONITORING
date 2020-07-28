from flask import Flask, render_template, Response,request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import cv2
import socket 
import io 
import imutils
import os
import numpy as np
from scipy.spatial import distance as dist

from playsound import playsound
from threading import Thread
rt=[]


net=cv2.dnn.readNetFromDarknet('yolo-coco\yolov3-tiny.cfg','yolo-coco\yolov3-tiny.weights')
labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
###################################################
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#determine only the output layer names that we need from yolo
ln=net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#print(ln)
def detect_person(frame,net,ln,personIdx=0):
    (H,W)=frame.shape[:2]
    results=[]
    # construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
    blob=cv2.dnn.blobFromImage(frame,1/255.0,(416,416),swapRB=True,crop=False)
    net.setInput(blob)
    layerOutputs=net.forward(ln)
    # initialize our lists of detected bounding boxes, centroids, and
	# confidences, respectively
    boxes=[]
    centroids=[]
    confidences=[]
    #loop over each of layer outputs
    for output in layerOutputs:
        #loop over each detection
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
			# of the current object detection
            scores=detection[5:]
            classID=np.argmax(scores)
            confidence=scores[classID]
            # filter detections by (1) ensuring that the object
			# detected was a person and (2) that the minimum
			# confidence is met
            if classID == personIdx and confidence>0.01:
                # scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX,centerY,width,height)=box.astype("int")
                # use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				
                X = int(centerX-(width/2))
                y = int(centerY-(height/2))
				# update our list of bounding box coordinates,
				# centroids, and confidences
                boxes.append([X,y,int(width),int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
                #apply NMS to supress weak overlapping bounding boxes
        idxs=cv2.dnn.NMSBoxes(boxes,confidences,0.01,0.1)
        #ensure that atleast one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # update our results list to consist of the person
                # # prediction probability, bounding box coordinates,
                # # and the centroid
                r = (confidences[i], (x, y, x + w, y + h), centroids[i])
                results.append(r)
    # return the list of results
    return results
app = Flask(__name__) 
x='' 
def plays(c):
    while c!=0:
        playsound('static\Sounds\swiftly.mp3')
        c=c-1
def gen(x):
   vb=0
   if x=='0':
       vc= cv2.VideoCapture(0)
   else:
       vc=cv2.VideoCapture(x)

     
   while True :
       rval,frame=vc.read()
       if not rval :
           break
       frame=imutils.resize(frame,width=700)
       if cv2.waitKey(1) & 0xFF ==ord('q') :
           break
       results=detect_person(frame,net,ln,personIdx=LABELS.index("person"))
       violate=set()
       #print(len(results))
       if len(results)>2:
           centroids=np.array([r[2] for r in results])
           D=dist.cdist(centroids,centroids,metric="euclidean")
           for i in range(0,D.shape[0]):
               for j in range(i+1,D.shape[1]):
                   if D[i,j]<150:
                       #t(D[i,j])
                       violate.add(i)
                       violate.add(j)
       for (i,(prob,bbox,centroid)) in enumerate(results):
           (startX,startY,endX,endY)=bbox
           (cX,cY)=centroid
           color=(0,255,0)
           if i in violate:
               vb=vb+1
               Thread(target=plays, args=(1,)).start()
               color=(0,0,255)
           cv2.rectangle(frame,(startX,startY),(endX,endY),color,2)
           cv2.circle(frame,(cX,cY),5,color,1)
       rt.append(len(violate))
       cv2.imwrite('pic.jpg',frame)
       yield(b'--frame\r\n'
              b'Content-Type :image/jpeg\r\n\r\n'+open('pic.jpg','rb').read()+b'\r\n'  )
   vc.release()
   #print(np.mean(rt))
   #print(np.max(rt))
   cv2.destroyAllWindows()
@app.route('/') 
def index(): 
   """Video streaming .""" 
   #print("hi")
   return render_template('index.html') 

   

 


#@app.route('/video_feed') 

@app.route("/forward/", methods=['POST'])
def video_feed(): 
    pass

@app.route("/webcam/", methods=['POST'])
def move_forward():

    return Response(gen('0'), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      #print(f.filename)
      x=str(f.filename)
      return Response(gen(x), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')
      return render_template('n.html')


    
if __name__ == '__main__': 
    app.run(debug=True, threaded=True)
