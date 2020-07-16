import face_recognition
import numpy as np
import cv2
from datetime import datetime
from datetime import date
from datetime import time
import time

import sys

video_capture = cv2.VideoCapture(0)

#video_capture.set(3,1920)
#video_capture.set(4,1080)


known_face_encodings = []
known_face_names = []
today = date.today()   
t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)



imgJeevan = face_recognition.load_image_file('/root/Desktop/Face-Detection/images/jeevan.jpg')
imgJeevanEncoding = face_recognition.face_encodings(imgJeevan)[0]

imgGundu = face_recognition.load_image_file('/root/Desktop/Face-Detection/images/gundu.jpg')
imgGunduEncoding = face_recognition.face_encodings(imgGundu)[0]

imgDhara = face_recognition.load_image_file('/root/Desktop/Face-Detection/images/dhara.jpg')
imgDharaEncoding = face_recognition.face_encodings(imgDhara)[0]

known_face_encodings = [imgJeevanEncoding , imgGunduEncoding , imgDharaEncoding]
known_face_names = ['Jeevan','Gundu' , 'Dhara']

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

i=0

while True:

    ret,frame = video_capture.read()

    small_frame = cv2.resize(frame,(0,0), fx=0.5,fy=0.5)

    rgb_small_frame = small_frame[:,::1]

    if process_this_frame:

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        name_list = []
        face_names = []

        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(known_face_encodings,face_encoding,tolerance=0.5)

            face_distances = face_recognition.face_distance(known_face_encodings,face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
              name = known_face_names[best_match_index]
              face_names.append(name)
    i+=1
    if i==5:
        curr_name = name
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        curr_time = current_time

        print(curr_name , today , curr_time , face_locations)
        original_stdout = sys.stdout 

        with open('filename.txt', 'a') as f:
         sys.stdout = f
         sys.stdout.close
         
         print(curr_name , today , curr_time , file=f )
         sys.stdout = original_stdout
         
        
          
          
    
        
       
        

    if len(face_names)==0:
        i=0

    process_this_frame = not process_this_frame                  
    
    for (top,right,bottom,left),name in zip(face_locations,face_names):
        top*=2
        right*=2
        bottom*=2
        left*=2

        cv2.rectangle(frame,(right,bottom),(left,top),(0,255,255),1)

        cv2.rectangle(frame,(left,bottom-35),(right,bottom),(0,255,0))
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame,name,(left+6,bottom-6),font,1.0,(255,255,255),1)
        #cv2.putText(frame,"hi",(right+23,bottom-36),font,1.0,(255,255,255),1)

    
    
    #cv2.namedWindow('Video',cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Video', 1920,1080)
    cv2.imshow('Video',frame)
    
   


    if cv2.waitKey(1) & 0xFF == ord('e'):
        break


video_capture.release()
cv2.destroyAllWindows()




