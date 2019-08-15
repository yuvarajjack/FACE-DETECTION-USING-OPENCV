import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person_name": 1}
with open("labels,pickle", "rb") as f:
    original_labels = pickle.load(f)
    labels = {v:k for k,v in original_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
    	print(x, y, w, h)
	#to find region of interest and save
   	roi_gray = gray[y:y+h, x:x+w]	#ycord_start,ycord_end
	roi_color = frame[y:y+h, x:x+w]

        id, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            print(id)
            print(labels[id])
            font = cv2.FONT_HERSHEY_PLAIN
            name = labels[id]                  # giving name  to image
            color =(255, 0, 0)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

    	output_image = "my-image.png"
    	cv2.imwrite(output_image,roi_gray)

	#to make a frame
	color = (0, 255, 0) #BGR 0-255
	stroke = 2
	width = x + w		#width
	height = y + h		#height
	cv2.rectangle(frame, (x, y), (width, height), color, stroke)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('gray',gray)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
