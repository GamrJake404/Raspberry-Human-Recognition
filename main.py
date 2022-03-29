import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import mediapipe as mp 
import os


mpHands = mp.solutions.hands 
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.1)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    body_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_fullbody.xml")
    faces = face_cascade.detectMultiScale(gray, 2, 5)
    bodies = face_cascade.detectMultiScale(gray, 1.4, 5)

    imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    hand_result = hands.process(imgRGB)
    results = pose.process(RGB)
    mp_drawing.draw_landmarks(
      frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    

    for (x, y, width, height) in faces:
        for face in faces:
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0,0,255), 2) 
                faces = frame[y:y + height, x:x + width]
                cv2.imwrite('face.jpg', faces)

    multiLandMarks = hand_result.multi_hand_landmarks
    if multiLandMarks:
        handPoints = []
        for handLms in multiLandMarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            for idx, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                handPoints.append((cx, cy))

        for point in handPoints:
            cv2.circle(frame, point, 5, (0,0,255), cv2.FILLED)

        fingersState = [0,0,0,0,0]

        if handPoints[8][1] < handPoints[6][1]:
            fingersState[0] = 1
        if handPoints[12][1] < handPoints[10][1]:
            fingersState[1] = 1
        if handPoints[16][1] < handPoints[14][1]:
            fingersState[2] = 1
        if handPoints[20][1] < handPoints[18][1]:
            fingersState[3] = 1
        if handPoints[4][0] > handPoints[2][0]:
            fingersState[4] = 1

        os.system("cls")
        if fingersState[0] == 1:
            print("index")
        if fingersState[1] == 1:
            print("middle")
        if fingersState[2] == 1:
            print("ring")
        if fingersState[3] == 1:
            print("pinky")
        if fingersState[4] == 1: 
            print("thumb")


    if cv2.waitKey(1) == ord(' '):
        break

    cv2.flip(frame, 1)
    cv2.imshow('Webcam', frame)

cap.release()
cv2.destroyAllWindows()
