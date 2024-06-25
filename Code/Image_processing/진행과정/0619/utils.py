import cv2
import numpy as np


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
        np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def draw_highlighted_text(image, text, position, font, scale, color, thickness, bg_color):
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    text_w, text_h = text_size
    x, y = position
    cv2.rectangle(image, (x, y - text_h - 10),
                  (x + text_w, y + 10), bg_color, -1)
    cv2.putText(image, text, position, font, scale,
                color, thickness, cv2.LINE_AA)


def draw_angle_arc(image, point1, point2, point3, color, w, h, reverse=False):
    center = tuple(np.multiply(point2, [w, h]).astype(int))
    start_point = tuple(np.multiply(point1, [w, h]).astype(int))
    end_point = tuple(np.multiply(point3, [w, h]).astype(int))

    radius = 30  # Fixed radius for small arc
    angle_start = int(np.degrees(np.arctan2(
        start_point[1] - center[1], start_point[0] - center[0])))
    angle_end = int(np.degrees(np.arctan2(
        end_point[1] - center[1], end_point[0] - center[0])))

    if reverse:
        angle_start, angle_end = angle_end, angle_start

    if angle_start > angle_end:
        angle_start, angle_end = angle_end, angle_start

    cv2.ellipse(image, center, (radius, radius), 0,
                angle_start, angle_end, color, 2)


def angle_to_percent(angle, min_angle=170, max_angle=65):
    return np.clip((min_angle - angle) / (min_angle - max_angle) * 100, 0, 100)
