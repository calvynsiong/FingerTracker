import cv2 as cv
import mediapipe as mp
import os
import time
import trackerModule as tm


cap = cv.VideoCapture(0)
# wCam, hCam = 640, 480
# cap.set(3, wCam)
# cap.set(4, hCam)

# Access all Image as a list
myList = os.listdir("./FingerImages")
# print(myList)
# Creating an overlay on the videos
overlayList = []


# Accesing detector class
detector = tm.Detector(detectionCon=0.75)


for i, imgPath in enumerate(myList):
    image = cv.imread(f"./FingerImages/{imgPath}")
    image = cv.resize(image, (200, 200))
    overlayList.append(image)


# Playing video

currTime = 0
prevTime = 0

tipIds = [4, 8, 12, 16, 20]
tipIds2 = [4, 8, 12, 16, 20]


while True:
    isPlaying, frame = cap.read()

    # frame[0:200, 0:200] = overlayList[0]

    frame = detector.findHands(frame)
    landmarkList = detector.findPosition(frame)
    # print(landmarkList)

    fingers = []

    if landmarkList:

        # Checks if y coordinate of tip is below (Also only compatible with right hand)
        fingers = [1 if (landmarkList[4][1] > landmarkList[3][1]) else 0]
        fingers += [1 if landmarkList[tipIds[id]][2] <
                    landmarkList[tipIds[id]-2][2] else 0 for id in range(1, 5)]
        # Second hand check
        if len(landmarkList) > 21:
            fingers += [1 if (landmarkList[24][1] >
                              landmarkList[23][1]) else 0]
            fingers += [1 if landmarkList[tipIds2[id]+20][2] <
                        landmarkList[tipIds2[id]+18][2] else 0 for id in range(1, 5)]
    print(fingers)

    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    cv.putText(frame, "Start with right hand first", (20, 50),
               cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
    cv.putText(frame, "(Yes I know second hand is broken)", (20, 80),
               cv.FONT_HERSHEY_PLAIN, .7, (255, 0, 0), 1)
    cv.putText(frame, f"FPS:{int(fps)}", (440, 50),
               cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    cv.putText(frame, f"Fingers Open: {len([i for i in fingers if i==1])}", (440, 100),
               cv.FONT_HERSHEY_PLAIN, 1.3, (255, 0, 0), 1)
    cv.imshow("Video", frame)

    if cv.waitKey(1) & 0xFF == ord("d"):
        break
