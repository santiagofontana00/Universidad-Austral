import cv2
import numpy as np
from contour import get_contours, filter_contours_by_area, compare_contours
from saved_contours import get_saved_contour
from frame_editor import apply_color_convertion, threshold, denoise, draw_contours



#crear trackbar
def create_trackbar(trackbar_name, window_name, slider_max):
    cv2.createTrackbar(trackbar_name, window_name, 0, slider_max, on_trackbar)


def on_trackbar(val):
    pass


def get_trackbar_value(trackbar_name, window_name):
    return int(cv2.getTrackbarPos(trackbar_name, window_name))


WINDOW_NAME = 'WINDOW'
BINARY_WINDOW = 'binary'

TRACKBAR_THRESH_NAME = 'Threshold'
TRACKBAR_THRESH_SLIDER_MAX = 255

TRACKBAR_KERNEL_NAME = 'Kernel size'
TRACKBAR_KERNEL_SLIDER_MAX = 20

trackbar_triangle_tol_name = 'Triangle Tolerance'
triangle_tol_max = 100

trackbar_square_tol_name = 'Square Tolerance'
square_tol_max = 100

trackbar_star_tol_name = 'Star Tolerance'
star_tol_max = 100

trackbar_min_area_name = 'Min Area'
contour_min_area_max = 10000


trackbar_max_area_name = 'Max Area'
contour_max_area_max = 99999

# BGR
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLUE = (255, 0, 0)

#contornos guardados
TRIANGLE_CONTOUR = get_saved_contour('tp_deteccion/figures/triangle.png')
SQUARE_CONTOUR = get_saved_contour('tp_deteccion/figures/square.png')
STAR_CONTOUR = get_saved_contour('tp_deteccion/figures/star.jpeg')



def main():
    cv2.namedWindow(WINDOW_NAME)
    cv2.namedWindow(BINARY_WINDOW)
    create_trackbar(TRACKBAR_THRESH_NAME, BINARY_WINDOW, TRACKBAR_THRESH_SLIDER_MAX)
    create_trackbar(TRACKBAR_KERNEL_NAME, BINARY_WINDOW, TRACKBAR_KERNEL_SLIDER_MAX)
    create_trackbar(trackbar_min_area_name, BINARY_WINDOW, contour_min_area_max)
    create_trackbar(trackbar_max_area_name, BINARY_WINDOW, contour_max_area_max)
    create_trackbar(trackbar_triangle_tol_name, WINDOW_NAME, triangle_tol_max)
    create_trackbar(trackbar_square_tol_name, WINDOW_NAME, square_tol_max)
    create_trackbar(trackbar_star_tol_name, WINDOW_NAME, star_tol_max)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        thresh_value = get_trackbar_value(TRACKBAR_THRESH_NAME, BINARY_WINDOW)
        kernel_size = get_trackbar_value(TRACKBAR_KERNEL_NAME, BINARY_WINDOW)
        kernel_size = max(1, kernel_size)  # Ensure kernel size is at least 1
        triangle_tol = get_trackbar_value(trackbar_triangle_tol_name, WINDOW_NAME)/100
        square_tol = get_trackbar_value(trackbar_square_tol_name, WINDOW_NAME)/100
        star_tol = get_trackbar_value(trackbar_star_tol_name, WINDOW_NAME)/100
        min_area = get_trackbar_value(trackbar_min_area_name, BINARY_WINDOW)
        max_area = get_trackbar_value(trackbar_max_area_name, BINARY_WINDOW)
        
        #pasar a escala de grises y despues a binario
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh1 = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)

        # Perform denoising using opening and closing
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        #obtener los contornos
        contours = get_contours(frame=closing, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        #filtrar los contornos por area
        filtered_contours = filter_contours_by_area(contours=contours, min_area=min_area, max_area=max_area)

        for cont in filtered_contours:

            if compare_contours(cont, TRIANGLE_CONTOUR, triangle_tol):
                draw_contours(frame=frame, contours=[cont], color=COLOR_GREEN, thickness=3)
                x, y, _, _ = cv2.boundingRect(cont)
                cv2.putText(frame, "Triangle", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN, 2)

            elif compare_contours(cont, SQUARE_CONTOUR, square_tol):
                draw_contours(frame=frame, contours=[cont], color=COLOR_BLUE, thickness=3)
                x, y, _, _ = cv2.boundingRect(cont)
                cv2.putText(frame, "Square", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLUE, 2)

            elif compare_contours(cont, STAR_CONTOUR, star_tol):
                draw_contours(frame=frame, contours=[cont], color=COLOR_WHITE, thickness=3)
                x, y, _, _ = cv2.boundingRect(cont)
                cv2.putText(frame, "Star", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 2)

            else:
                draw_contours(frame=frame, contours=[cont], color=COLOR_RED, thickness=3)
                x, y, _, _ = cv2.boundingRect(cont)
                cv2.putText(frame, "Unknown", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, 2)

        #mostrar la imagen y terminar el programa con la tecla q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow(WINDOW_NAME, frame)
        cv2.imshow(BINARY_WINDOW, closing)

    cap.release()
    cv2.destroyAllWindows()


main()