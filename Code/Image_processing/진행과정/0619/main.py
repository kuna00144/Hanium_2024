import cv2
import mediapipe as mp
import numpy as np
from utils import calculate_angle, draw_highlighted_text, draw_angle_arc, angle_to_percent

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
count_left = 0
count_right = 0
dir_left = 0
dir_right = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Mirror the image
        frame = cv2.flip(frame, 1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Left arm
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Right arm
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angles
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_angle(
                right_shoulder, right_elbow, right_wrist)

            # Convert landmark positions to pixel coordinates
            h, w, _ = image.shape
            left_elbow_px = np.multiply(left_elbow, [w, h]).astype(int)
            right_elbow_px = np.multiply(right_elbow, [w, h]).astype(int)
            left_shoulder_px = np.multiply(left_shoulder, [w, h]).astype(int)
            left_wrist_px = np.multiply(left_wrist, [w, h]).astype(int)
            right_shoulder_px = np.multiply(right_shoulder, [w, h]).astype(int)
            right_wrist_px = np.multiply(right_wrist, [w, h]).astype(int)

            # Display angles with slight offset
            offset = 40
            left_text_position = (
                left_elbow_px[0] - offset, left_elbow_px[1] - offset)
            right_text_position = (
                right_elbow_px[0] - offset, right_elbow_px[1] - offset)

            # Highlight the text by adding a background rectangle
            draw_highlighted_text(image, str(int(left_angle)), left_text_position,
                                  cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, (0, 0, 0))
            draw_highlighted_text(image, str(int(right_angle)), right_text_position,
                                  cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, (0, 0, 0))

            # Draw arcs
            draw_angle_arc(image, left_shoulder, left_elbow,
                           left_wrist, (0, 255, 0), w, h)
            draw_angle_arc(image, right_shoulder, right_elbow,
                           right_wrist, (0, 255, 0), w, h, reverse=False)

            # Emphasize arm joints
            for point in [left_shoulder_px, left_elbow_px, left_wrist_px, right_shoulder_px, right_elbow_px, right_wrist_px]:
                cv2.circle(image, tuple(point), 15, (0, 0, 255), cv2.FILLED)
                cv2.circle(image, tuple(point), 20, (0, 0, 255), 2)

            # Calculate bar height based on angle
            left_percent = angle_to_percent(left_angle)
            right_percent = angle_to_percent(right_angle)

            # 170도(팔 쭉 펴서) 찍어야 카운트 되는 코드
            """# Update count based on left_percent
            if left_percent == 100:
                if dir_left == 0:
                    count_left += 1
                    dir_left = 1
            if left_percent == 0:
                if dir_left == 1:
                    dir_left = 0

            # Update count based on right_percent
            if right_percent == 100:
                if dir_right == 0:
                    count_right += 1
                    dir_right = 1
            if right_percent == 0:
                if dir_right == 1:
                    dir_right = 0 """
            
            
            # 170도 안찍어도 카운트 되는 코드
            # Update count based on left_percent
            if left_percent == 100:
                if dir_left == 0:
                    count_left += 1
                    dir_left = 1
            else:
                dir_left = 0

            # Update count based on right_percent
            if right_percent == 100:
                if dir_right == 0:
                    count_right += 1
                    dir_right = 1
            else:
                dir_right = 0

            # Draw bar for left arm
            bar_x_start = int(w / 6)
            bar_x_end = int(w * 5 / 6)
            bar_y_start_left = int(h * 0.85)
            bar_y_end_left = bar_y_start_left + 25
            bar_y_start_right = bar_y_start_left + 35
            bar_y_end_right = bar_y_start_right + 25

            left_bar_width = int((left_percent / 100) *
                                 (bar_x_end - bar_x_start)) + bar_x_start
            right_bar_width = int((right_percent / 100) *
                                  (bar_x_end - bar_x_start)) + bar_x_start

            bar_color_left = (
                255, 0, 255) if left_percent == 100 else (0, 255, 0)
            bar_color_right = (
                255, 0, 255) if right_percent == 100 else (0, 255, 0)

            # Left bar(좌우반전 해서 right)
            cv2.rectangle(image, (bar_x_start, bar_y_start_left),
                          (bar_x_end, bar_y_end_left), (255, 255, 255), 2)
            cv2.rectangle(image, (bar_x_start, bar_y_start_left),
                          (left_bar_width, bar_y_end_left), bar_color_left, cv2.FILLED)
            cv2.putText(image, f'{int(left_percent)} %', (left_bar_width + 10, bar_y_end_left - 10),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(image, 'Right', (bar_x_start - 60, bar_y_end_left - 10),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

            # Right bar(좌우반전 해서 left)
            cv2.rectangle(image, (bar_x_start, bar_y_start_right),
                          (bar_x_end, bar_y_end_right), (255, 255, 255), 2)
            cv2.rectangle(image, (bar_x_start, bar_y_start_right),
                          (right_bar_width, bar_y_end_right), bar_color_right, cv2.FILLED)
            cv2.putText(image, f'{int(right_percent)} %', (right_bar_width + 10, bar_y_end_right - 10),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(image, 'Left', (bar_x_start - 60, bar_y_end_right - 10),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

            # Draw curl count
            cv2.rectangle(image, (10, 30), (450, 170), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, str(int(count_left)), (120, 100),
                        cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 10)
            cv2.putText(image, 'Right Count', (40, 150), # 좌우반전 했기때문에 right
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

            cv2.putText(image, str(int(count_right)), (320, 100),
                        cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 10)
            cv2.putText(image, 'Left Count', (260, 150),  # 좌우반전 했기때문에 left
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

        except Exception as e:
            print(e)
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(255, 255, 255), thickness=3, circle_radius=3),
                                  mp_drawing.DrawingSpec(
                                      color=(255, 255, 0), thickness=3, circle_radius=3)
                                  )
        cv2.imshow('Mediapipe Feed', image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
