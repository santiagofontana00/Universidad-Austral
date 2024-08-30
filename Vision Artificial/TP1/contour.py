import cv2


def get_contours(frame, mode, method):
    contours, hierarchy = cv2.findContours(frame, mode, method)
    return contours


def filter_contours_by_area(contours, min_area, max_area):
    filtered_contours = []
    for cnt in contours:
        if min_area <= cv2.contourArea(cnt) <= max_area:
            filtered_contours.append(cnt)
    return filtered_contours


def compare_contours(contour_to_compare, saved_contour, max_diff):
    return cv2.matchShapes(contour_to_compare, saved_contour, cv2.CONTOURS_MATCH_I1, 0.0) < max_diff


def get_bounding_rect(contour):
    return cv2.boundingRect(contour)