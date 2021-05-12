import cv2 as cv
import mediapipe as mp
import time


def main():


        # Capturing video with webcam
    cap = cv.VideoCapture(0)


    # Creating object
    mpHands = mp.solutions.hands
    # There are 4 parameters to hands (static_image_mode=False(ifTrue it will track all the time), max_num_hands=2, min_detection_confidence=.5, min_tracking_confidence=.5   )
    # When tracking is below confidence range, detection restarts
    hands = mpHands.Hands()
    # Drawing over hands
    mpDraw = mp.solutions.drawing_utils

    # Setting frame rate
    prevTime = 0
    currTime = 0


    # Capture video
    while True:
        isCaptured, frame = cap.read()
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # Process rgb frames for hands
        results = hands.process(frameRGB)
        # print(results.multi_hand_landmarks)

        # If there is a hand detected (no landmark == None)
        fingers = 0
        if results.multi_hand_landmarks:
            # For each hand detected , draw the landmark points  (hand points) on the frame, while connecting them
            for handLandmark in results.multi_hand_landmarks:
                # List index and info of handLandmark.landmark (which has x,y,z coordinates) + lm will be x,y,z coordinates
                # The index provides information about which landmark point it is
                for id, lm in enumerate(handLandmark.landmark):
                    # Obtaining height,width and channel of image
                    h, w, channel = frame.shape
                    # Converting x and y coordinates of landmark properlyd
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    # Checks for point indexes and draws circles around them if they match
                    if id in [4, 8, 12, 16, 20]:
                        cv.circle(frame, (cx, cy), 7, (0, 0, 0), -1)
                        fingers += 1

                mpDraw.draw_landmarks(frame, handLandmark,
                                    mpHands.HAND_CONNECTIONS)

        # Listing fps (1 frame/ per elasped time)
        currTime = time.time()
        fps = 1/(currTime-prevTime)
        prevTime = currTime

        cv.putText(frame, f"Frame Rate:{str(int(fps))}", (0, 70),
                cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)

        cv.imshow("Video", frame)
        # # Stops the video when 20s reach or when letter d is pressed
        if cv.waitKey(1) & 0xFF == ord("d"):
            break

if __name__ == "__main__":
    main()
