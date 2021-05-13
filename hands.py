import cv2 as cv
import mediapipe as mp
import time
import trackerModule as tm

# PRESS d on keyboard to stop video

# Capturing video with webcam
cap = cv.VideoCapture(0)
# Initializing class
detector = tm.Detector()

# Setting frame rate
prevTime = 0
currTime = 0

# Capture video
while True:
    isCaptured, frame = cap.read()
    frame = detector.findHands(frame)
    landmarkList = detector.findPosition(frame)

    # If landmarkList is not empty
    if (landmarkList):
        # Provides position of specific landmark
        print(landmarkList[0])

        # Listing fps (1 frame/ per elasped time)
    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    cv.putText(frame, f"Frame Rate:{str(int(fps))}", (0, 70),
               cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)

    cv.imshow("Video", frame)
    # # Stops the video when 20s reach or when letter d is pressed
    # bitwise and evaluates cv.waitkey and if the second statement is true, it finally breaks it
    if cv.waitKey(1) & 0xFF == ord("d"):
        break
