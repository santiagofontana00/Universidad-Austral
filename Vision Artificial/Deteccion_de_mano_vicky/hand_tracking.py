# COMPUTER VISION PROJECTS

# HAND-TRACKING

import cv2
import mediapipe as mp
import time

# STEP 0: create video object, runs webcam
cap = cv2.VideoCapture(0)   # number = number of webcam


# STEP 1: create module for hand
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw =  mp.solutions.drawing_utils #draws lines btw points

# STEP BLEH: framerate
pTime = 0
cTime = 0

# STEP 0: continuation
while True:
	success, img = cap.read()
	
	# Voltear la imagen horizontalmente
	img = cv2.flip(img, 1)

	# STEP 2: send in rgb image
	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = hands.process(imgRGB)
	#print(results.multi_hand_landmarks) #prints if obj detected

	#STEP 3: extract desired object (when multiple objs present)
	if results.multi_hand_landmarks:
		for handLms in results.multi_hand_landmarks:

			#step flah: get info from hands
			for id, lm in enumerate(handLms.landmark):
				#print(id, lm)
				h, w, c = img.shape
				cx, cy = int(lm.x*w), int(lm.y*h)
				print(id, cx, cy)

				if id == 4:
					cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

			mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) #actually, this last thing is what connects the points

	# STEP BLEH: FPS
	cTime = time.time()
	fps = 1/(cTime-pTime)
	pTime = cTime

	cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)


	#STEP X: display
	cv2.imshow("Image", img)
	cv2.waitKey(1)


