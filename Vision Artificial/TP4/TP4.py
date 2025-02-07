import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Importing all images
# Preguntar al usuario qué cancha quiere usar
print("En qué cancha querés jugar?")
print("1. Boca")
print("2. River") 
print("3. Newell's")
cancha = input("Ingresa el número de tu elección (1-3): ")

if cancha == "1":
    imgBackground = cv2.imread("Universidad-Austral/Vision Artificial/TP4/canchaBoca.jpeg")
    imgBackground = cv2.resize(imgBackground, (1280, 720))
elif cancha == "2":
    imgBackground = cv2.imread("Universidad-Austral/Vision Artificial/TP4/canchaRiver.jpg")
    imgBackground = cv2.resize(imgBackground, (1280, 720))
else:
    imgBackground = cv2.imread("Universidad-Austral/Vision Artificial/TP4/canchaNewells.jpg")
    imgBackground = cv2.resize(imgBackground, (1280, 720))

imgGameOver = cv2.imread("Universidad-Austral/Vision Artificial/TP4/gameOver.png")

# Primero leemos Ball.png para obtener las dimensiones de referencia
bat_reference = cv2.imread("Universidad-Austral/Vision Artificial/TP4/Ball.png", cv2.IMREAD_UNCHANGED)
ball_height, ball_width = bat_reference.shape[:2]


# Usar pelota por default
imgBall = cv2.imread("Universidad-Austral/Vision Artificial/TP4/Ball2.png", cv2.IMREAD_UNCHANGED)

imgBall = cv2.resize(imgBall, (ball_width, ball_height))

# Preguntar al usuario qué equipo quiere para el jugador izquierdo
print("\nCon qué equipo quiere jugar el jugador de la izquierda?")
print("1. Boca")
print("2. River")
print("3. Newell's")
equipo1 = input("Ingresa el número de tu elección (1-3): ")

# Primero leemos bat1.png para obtener las dimensiones de referencia
bat_reference = cv2.imread("Universidad-Austral/Vision Artificial/TP4/bat1.png", cv2.IMREAD_UNCHANGED)
bat_height, bat_width = bat_reference.shape[:2]



if equipo1 == "1":
    imgBat1 = cv2.imread("Universidad-Austral/Vision Artificial/TP4/boca.jpg")
    imgBat1 = cv2.cvtColor(imgBat1, cv2.COLOR_BGR2BGRA)
    imgBat1 = cv2.resize(imgBat1, (bat_width, bat_height))
elif equipo1 == "2":
    imgBat1 = cv2.imread("Universidad-Austral/Vision Artificial/TP4/river.jpg")
    imgBat1 = cv2.cvtColor(imgBat1, cv2.COLOR_BGR2BGRA)
    imgBat1 = cv2.resize(imgBat1, (bat_width, bat_height))
else:
    imgBat1 = cv2.imread("Universidad-Austral/Vision Artificial/TP4/newells.jpg")
    imgBat1 = cv2.cvtColor(imgBat1, cv2.COLOR_BGR2BGRA)
    imgBat1 = cv2.resize(imgBat1, (bat_width, bat_height))

# Preguntar al usuario qué equipo quiere para el jugador derecho
print("\nCon qué equipo quiere jugar el jugador de la derecha?")
print("1. Boca")
print("2. River")
print("3. Newell's")
equipo2 = input("Ingresa el número de tu elección (1-3): ")

if equipo2 == "1":
    imgBat2 = cv2.imread("Universidad-Austral/Vision Artificial/TP4/boca.jpg")
    imgBat2 = cv2.cvtColor(imgBat2, cv2.COLOR_BGR2BGRA)
    imgBat2 = cv2.resize(imgBat2, (bat_width, bat_height))
elif equipo2 == "2":
    imgBat2 = cv2.imread("Universidad-Austral/Vision Artificial/TP4/river.jpg")
    imgBat2 = cv2.cvtColor(imgBat2, cv2.COLOR_BGR2BGRA)
    imgBat2 = cv2.resize(imgBat2, (bat_width, bat_height))
else:
    imgBat2 = cv2.imread("Universidad-Austral/Vision Artificial/TP4/newells.jpg")
    imgBat2 = cv2.cvtColor(imgBat2, cv2.COLOR_BGR2BGRA)
    imgBat2 = cv2.resize(imgBat2, (bat_width, bat_height))

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Variables
ballPos = [100, 100]
speedX = 15
speedY = 15
gameOver = False
score = [0, 0]

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    imgRaw = img.copy()

    # Find the hand and its landmarks
    hands, img = detector.findHands(img, flipType=False)  # with draw

    # Overlaying the background image
    img = cv2.addWeighted(img, 0.2, imgBackground, 0.8, 0)

    # Check for hands
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = imgBat1.shape
            y1 = y
            y1 = np.clip(y1, 20, 415)

            if hand['type'] == "Left":
                img = cvzone.overlayPNG(img, imgBat1, (59, y1))
                if 59 < ballPos[0] < 59 + w1 and y1  < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] += 30
                    score[0] += 1

            if hand['type'] == "Right":
                img = cvzone.overlayPNG(img, imgBat2, (1195, y1))
                if 1195 - 50 < ballPos[0] < 1195 and y1  < ballPos[1] < y1 + h1 :
                    speedX = -speedX
                    ballPos[0] -= 30
                    score[1] += 1

    # Game Over
    if ballPos[0] < 40 or ballPos[0] > 1200:
        gameOver = True

    if gameOver:
        img = imgGameOver
        cv2.putText(img, str(score[1] + score[0]).zfill(2), (585, 360), cv2.FONT_HERSHEY_COMPLEX,
                    2.5, (200, 0, 200), 5)

    # If game not over move the ball
    else:

        # Move the Ball
        if ballPos[1] >= 500 or ballPos[1] <= 10:
            speedY = -speedY

        ballPos[0] += speedX
        ballPos[1] += speedY

        # Draw the ball
        img = cvzone.overlayPNG(img, imgBall, ballPos)

        cv2.putText(img, str(score[0]), (300, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
        cv2.putText(img, str(score[1]), (900, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)

    img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120))

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        ballPos = [100, 100]
        speedX = 15
        speedY = 15
        gameOver = False
        score = [0, 0]
        imgGameOver = cv2.imread("Universidad-Austral/Vision Artificial/TP4/gameOver.png")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
