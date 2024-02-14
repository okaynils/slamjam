import numpy as np
import cv2

def calc_keypoints(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    frame = cv2.drawKeypoints(frame, keypoints, None)
    return frame

video = cv2.VideoCapture('footage/driving.mp4')

if (video.isOpened()== False): 
    print("Error opening video file") 

while(video.isOpened()): 
    ret, frame = video.read() 
    if ret == True:
        k_ = calc_keypoints(frame)
        cv2.imshow('Frame', frame+k_)
        if cv2.waitKey(25) & 0xFF == ord('q'): # Press Q on keyboard to exit 
            break
    else: 
        break

video.release() 

cv2.destroyAllWindows()