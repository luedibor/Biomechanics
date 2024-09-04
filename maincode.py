import cv2
import mediapipe as mp
import os
import math
import time
import numpy as np
import pandas as pd

# Declaración de la lista para almacenar los datos
datos = []

def draw_landmarks(image, results, angle_point_left, angle_point_right, shoulder_point_left, shoulder_point_right):
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    if angle_point_left is not None:
        cv2.circle(image, angle_point_left, 5, (255, 0, 255), -1)

    if angle_point_right is not None:
        cv2.circle(image, angle_point_right, 5, (255, 0, 255), -1)

    if shoulder_point_left is not None:
        cv2.circle(image, shoulder_point_left, 5, (255, 0, 255), -1)

    if shoulder_point_right is not None:
        cv2.circle(image, shoulder_point_right, 5, (255, 0, 255), -1)

def calculate_angle(a, b, c, angle_type='external'):
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(math.degrees(radians))

    if angle_type == 'internal':
        angle = 360 - angle

    return angle

def calculate_hip_knee_ankle_angle(results, side, width, height, angle_type='external'):
    pose_landmarks = results.pose_landmarks.landmark
    hip_landmark, knee_landmark, ankle_landmark = None, None, None

    if side == 'left':
        hip_landmark = pose_landmarks[mp.solutions.holistic.PoseLandmark.LEFT_HIP.value]
        knee_landmark = pose_landmarks[mp.solutions.holistic.PoseLandmark.LEFT_KNEE.value]
        ankle_landmark = pose_landmarks[mp.solutions.holistic.PoseLandmark.LEFT_ANKLE.value]
    elif side == 'right':
        hip_landmark = pose_landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_HIP.value]
        knee_landmark = pose_landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_KNEE.value]
        ankle_landmark = pose_landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_ANKLE.value]

    hip_coords = [hip_landmark.x, hip_landmark.y]
    knee_coords = [knee_landmark.x, knee_landmark.y]
    ankle_coords = [ankle_landmark.x, ankle_landmark.y]

    angle = calculate_angle(hip_coords, knee_coords, ankle_coords, angle_type)

    knee_point = (int(knee_coords[0] * width), int(knee_coords[1] * height))
    return angle, knee_point

def calculate_shoulder_elbow_wrist_angle(results, side, width, height, angle_type='external'):
    pose_landmarks = results.pose_landmarks.landmark
    shoulder_landmark, elbow_landmark, wrist_landmark = None, None, None

    if side == 'left':
        shoulder_landmark = pose_landmarks[mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER.value]
        elbow_landmark = pose_landmarks[mp.solutions.holistic.PoseLandmark.LEFT_ELBOW.value]
        wrist_landmark = pose_landmarks[mp.solutions.holistic.PoseLandmark.LEFT_WRIST.value]
    elif side == 'right':
        shoulder_landmark = pose_landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER.value]
        elbow_landmark = pose_landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_ELBOW.value]
        wrist_landmark = pose_landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_WRIST.value]

    shoulder_coords = [shoulder_landmark.x, shoulder_landmark.y]
    elbow_coords = [elbow_landmark.x, elbow_landmark.y]
    wrist_coords = [wrist_landmark.x, wrist_landmark.y]

    angle = calculate_angle(shoulder_coords, elbow_coords, wrist_coords, angle_type)

    elbow_point = (int(elbow_coords[0] * width), int(elbow_coords[1] * height))
    return angle, elbow_point

def calculate_angular_velocities(angles, time_between_frames):
    return np.diff(angles) / time_between_frames

def calculate_angular_accelerations(angular_velocities, time_between_frames):
    return np.diff(angular_velocities) / time_between_frames

def show_video_with_landmarks(video_path, output_filename, angle_type='external'):
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    window_width = int(width * 0.6)
    window_height = int(height * 0.6)

    cv2.namedWindow('Video with Landmarks', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video with Landmarks', window_width, window_height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(os.path.join('videos_salida', output_filename), fourcc, 30, (window_width, window_height))

    start_time = time.time()
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            elapsed_time = time.time() - start_time

            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = holistic.process(image)

            angle_left, knee_point_left = calculate_hip_knee_ankle_angle(results, 'left', width, height, angle_type='internal')
            angle_right, knee_point_right = calculate_hip_knee_ankle_angle(results, 'right', width, height, angle_type='external')

            angle_shoulder_left, shoulder_point_left = calculate_shoulder_elbow_wrist_angle(results, 'left', width, height, angle_type='internal')
            angle_shoulder_right, shoulder_point_right = calculate_shoulder_elbow_wrist_angle(results, 'right', width, height, angle_type='external')

            draw_landmarks(frame, results, knee_point_left, knee_point_right, shoulder_point_left, shoulder_point_right)

            cv2.putText(frame, f'Angle Hip-Knee-Ankle-Left: {angle_left:.2f} degrees', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Angle Hip-Knee-Ankle-Right: {angle_right:.2f} degrees', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f'Angle Shoulder-Elbow-Left: {angle_shoulder_left:.2f} degrees', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Angle Shoulder-Elbow-Right: {angle_shoulder_right:.2f} degrees', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Almacenar datos en la lista
            datos.append({
                'Time': elapsed_time,
                'Angle_Hip_Knee_Left': angle_left,
                'Angle_Hip_Knee_Right': angle_right,
                'Angle_Shoulder_Elbow_Left': angle_shoulder_left,
                'Angle_Shoulder_Elbow_Right': angle_shoulder_right,
                'Angular_Velocity_Hip_Knee_Left': None,
                'Angular_Velocity_Hip_Knee_Right': None,
                'Angular_Acceleration_Hip_Knee_Left': None,
                'Angular_Acceleration_Hip_Knee_Right': None,
                'Angular_Velocity_Shoulder_Elbow_Left': None,
                'Angular_Velocity_Shoulder_Elbow_Right': None,
                'Angular_Acceleration_Shoulder_Elbow_Left': None,
                'Angular_Acceleration_Shoulder_Elbow_Right': None
            })

            # Imprimir datos en la consola
            print(f'Time: {elapsed_time:.2f} seconds')
            print(f'Angle Hip-Knee-Ankle-Left: {angle_left:.2f} degrees')
            print(f'Angle Hip-Knee-Ankle-Right: {angle_right:.2f} degrees')
            print(f'Angle Shoulder-Elbow-Left: {angle_shoulder_left:.2f} degrees')
            print(f'Angle Shoulder-Elbow-Right: {angle_shoulder_right:.2f} degrees')
            print('\n')

            output_video.write(cv2.resize(frame, (window_width, window_height)))
            cv2.imshow('Video with Landmarks', cv2.resize(frame, (window_width, window_height)))

            # Esperar a tener al menos dos ángulos 
            if len(datos) >= 2:
                # Obtener los dos últimos ángulos
                angle_left_prev = datos[-2]['Angle_Hip_Knee_Left']
                angle_right_prev = datos[-2]['Angle_Hip_Knee_Right']

                # Calcular el tiempo entre los dos últimos fotogramas
                tiempo_entre_fotogramas_prev = datos[-1]['Time'] - datos[-2]['Time']

                # Calcular velocidad angular de caderas y rodillas
                angular_velocity_left = (angle_left - angle_left_prev) / tiempo_entre_fotogramas_prev
                angular_velocity_right = (angle_right - angle_right_prev) / tiempo_entre_fotogramas_prev

                # Calcular velocidad angular de hombro y codo
                angular_velocity_shoulder_left = (angle_shoulder_left - datos[-2]['Angle_Shoulder_Elbow_Left']) / tiempo_entre_fotogramas_prev
                angular_velocity_shoulder_right = (angle_shoulder_right - datos[-2]['Angle_Shoulder_Elbow_Right']) / tiempo_entre_fotogramas_prev


                # Calcular aceleración angular de caderas y rodillas
                if datos[-2]['Angular_Velocity_Hip_Knee_Left'] is not None:
                    angular_acceleration_left = (angular_velocity_left - datos[-2]['Angular_Velocity_Hip_Knee_Left']) / tiempo_entre_fotogramas_prev
                else:
                    angular_acceleration_left = None

                if datos[-2]['Angular_Velocity_Hip_Knee_Right'] is not None:
                    angular_acceleration_right = (angular_velocity_right - datos[-2]['Angular_Velocity_Hip_Knee_Right']) / tiempo_entre_fotogramas_prev
                else:
                    angular_acceleration_right = None

                # Calcular aceleración angular de hombro y codo
                if datos[-2]['Angular_Velocity_Shoulder_Elbow_Left'] is not None:
                    angular_acceleration_shoulder_left = (angular_velocity_shoulder_left - datos[-2]['Angular_Velocity_Shoulder_Elbow_Left']) / tiempo_entre_fotogramas_prev
                else:
                    angular_acceleration_shoulder_left = None

                if datos[-2]['Angular_Velocity_Shoulder_Elbow_Right'] is not None:
                    angular_acceleration_shoulder_right = (angular_velocity_shoulder_right - datos[-2]['Angular_Velocity_Shoulder_Elbow_Right']) / tiempo_entre_fotogramas_prev
                else:
                    angular_acceleration_shoulder_right = None

                
                datos[-1]['Angular_Velocity_Hip_Knee_Left'] = angular_velocity_left
                datos[-1]['Angular_Velocity_Hip_Knee_Right'] = angular_velocity_right
                datos[-1]['Angular_Acceleration_Hip_Knee_Left'] = angular_acceleration_left
                datos[-1]['Angular_Acceleration_Hip_Knee_Right'] = angular_acceleration_right
                datos[-1]['Angular_Velocity_Shoulder_Elbow_Left'] = angular_velocity_shoulder_left
                datos[-1]['Angular_Velocity_Shoulder_Elbow_Right'] = angular_velocity_shoulder_right
                datos[-1]['Angular_Acceleration_Shoulder_Elbow_Left'] = angular_acceleration_shoulder_left
                datos[-1]['Angular_Acceleration_Shoulder_Elbow_Right'] = angular_acceleration_shoulder_right

                # Imprimir velocidades angulares y aceleraciones angulares
                print(f'Angular Velocity Hip-Knee-Ankle-Left: {angular_velocity_left:.2f} degrees per second')
                print(f'Angular Velocity Hip-Knee-Ankle-Right: {angular_velocity_right:.2f} degrees per second')
                
                if angular_acceleration_left is not None:
                    print(f'Angular Acceleration Hip-Knee-Ankle-Left: {angular_acceleration_left:.2f} degrees per second^2')
                else:
                    print('Angular Acceleration Hip-Knee-Ankle-Left: ')

                if angular_acceleration_right is not None:
                    print(f'Angular Acceleration Hip-Knee-Ankle-Right: {angular_acceleration_right:.2f} degrees per second^2')
                else:
                    print('Angular Acceleration Hip-Knee-Ankle-Right: ')

                # Imprimir velocidades angulares y aceleraciones angulares del hombro y el codo
                print(f'Angular Velocity Shoulder-Elbow-Left: {angular_velocity_shoulder_left:.2f} degrees per second')
                print(f'Angular Velocity Shoulder-Elbow-Right: {angular_velocity_shoulder_right:.2f} degrees per second')

                if angular_acceleration_shoulder_left is not None:
                    print(f'Angular Acceleration Shoulder-Elbow-Left: {angular_acceleration_shoulder_left:.2f} degrees per second^2')
                else:
                    print('Angular Acceleration Shoulder-Elbow-Left: N/A')

                if angular_acceleration_shoulder_right is not None:
                    print(f'Angular Acceleration Shoulder-Elbow-Right: {angular_acceleration_shoulder_right:.2f} degrees per second^2')
                else:
                    print('Angular Acceleration Shoulder-Elbow-Right: ')

    # Convertir la lista de datos en un DataFrame de Pandas
            df = pd.DataFrame(datos)

    # Guardar el DataFrame en un archivo CSV
            csv_filename = 'tabla.csv'
            df.to_csv(csv_filename, index=False)
            print(f"Archivo CSV guardado en: {os.path.abspath(csv_filename)}")
            

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    show_video_with_landmarks('subject1.mov', 'video_con_angulo_y_punto.mp4')
