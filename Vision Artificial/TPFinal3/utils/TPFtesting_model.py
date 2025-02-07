import cv2
import numpy as np
import glob
import math
from utils.TPFhu_moments_generation import hu_moments_of_file
from utils.TPFlabel_converters import int_to_label


def load_and_test_saved_images(model):
    files = glob.glob('./TPFinal3/databaseASL/TPFtesting/*')
    for f in files:
        hu_moments = hu_moments_of_file(f) # Genera los momentos de hu de los files de testing
        sample = np.array([hu_moments], dtype=np.float32) # numpy
        testResponse = model.predict(sample)[1] # Predice la clase de cada file

        #Lee la imagen y la imprime con un texto
        image = cv2.imread(f)
        image_with_text = cv2.putText(image, int_to_label(testResponse), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("result", image_with_text)
        cv2.waitKey(0)

def nothing(x):
    pass

def load_and_test_camera(model):
    WINDOW_NAME = 'Shape Detection'
    BINARY_WINDOW = 'Binary Image'
    COLOR_GREEN = (0, 255, 0)
    COLOR_BLUE = (255, 0, 0)

    cv2.namedWindow(WINDOW_NAME)
    cv2.namedWindow(BINARY_WINDOW)
    cv2.createTrackbar('Min Area', WINDOW_NAME, 100, 10000, nothing)
    cv2.createTrackbar('Max Area', WINDOW_NAME, 10000, 100000, nothing)
    cv2.createTrackbar('Threshold', WINDOW_NAME, 0, 255, nothing)

    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        threshold_value = cv2.getTrackbarPos('Threshold', WINDOW_NAME)
        _, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = cv2.getTrackbarPos('Min Area', WINDOW_NAME)
        max_area = cv2.getTrackbarPos('Max Area', WINDOW_NAME)

        if contours:
            filtered_contours = [cnt for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area]
            
            for shape_contour in filtered_contours:
                moments = cv2.moments(shape_contour)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                else:
                    cx, cy = 0, 0

                huMoments = cv2.HuMoments(moments)

                for i in range(7):
                    huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))

                sample = np.array([huMoments.flatten()], dtype=np.float32)
                _, predicted_class = model.predict(sample)
                
                label = int_to_label(predicted_class)
                
                # Draw contour
                cv2.drawContours(frame, [shape_contour], 0, COLOR_GREEN, 2)
                # Put label near the centroid of the shape
                cv2.putText(frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_BLUE, 2)

        cv2.imshow(WINDOW_NAME, frame)
        cv2.imshow(BINARY_WINDOW, binary)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

