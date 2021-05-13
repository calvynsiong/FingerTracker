import cv2 as cv
import mediapipe as mp
import time


# Creating a class for hand detector

class Detector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        # Creating object for hand tracking module
        self.mpHands = mp.solutions.hands
        # There are 4 parameters to hands (static_image_mode=False(ifTrue it will track all the time), max_num_hands=2, min_detection_confidence=.5, min_tracking_confidence=.5   )
        # When tracking is below confidence range, detection restarts
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, self.detectionCon, self.trackCon)
        # Drawing over handsd
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        self.frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # Process rgb frames for hands
        self.results = self.hands.process(self.frameRGB)
        # print(results.multi_hand_landmarks)

        # If there is a hand detected (no landmark == None)
        if self.results.multi_hand_landmarks:
            # For each hand detected , draw the landmark points  (hand points) on the frame, while connecting them
            for handLandmark in self.results.multi_hand_landmarks:
                # If draw is true, it will draw on landmark
                if draw:
                    self.mpDraw.draw_landmarks(
                        frame, handLandmark, self.mpHands.HAND_CONNECTIONS)
        # Returns the frame
        return frame

    def findPosition(self, frame, handNumber=0, draw=True):
        landmarkList = []
        if self.results.multi_hand_landmarks:
            # Access specific hand with handNumber
            myHand = self.results.multi_hand_landmarks[handNumber]
            # List index and info of handLandmark.landmark (which has x,y,z coordinates) + lm will be x,y,z coordinates
            # The index provides information about which landmark point it is
            for id, lm in enumerate(myHand.landmark):
                # Obtaining height,width and channel of image
                h, w, channel = frame.shape
                # Converting x and y coordinates of landmark properlyd
                cx, cy = int(lm.x*w), int(lm.y*h)
                # Append the location + id of the specific landmark points to the list
                landmarkList.append([id, cx, cy])
                # Checks for point indexes and draws circles around them if they match
                if id in [4, 8, 12, 16, 20] and draw:
                    cv.circle(frame, (cx, cy), 7, (71, 99, 255), -1)

        return landmarkList

            # To track 2 hands, replace myHand with ==> for myHand in self.results.multi_hand_landmarks:

def main():

    # Capturing video with webcam
    cap = cv.VideoCapture(0)
    # Initializing class
    detector = Detector()

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


if __name__ == "__main__":
    main()
