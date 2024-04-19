import cv2
import math
import numpy as np
import mediapipe as mp

import websocket
import threading

# print("open camera")




def get_user_input(ws):
    mp_face_mesh = mp.solutions.face_mesh


    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    else:
        print("open camera")

    count = 0
    direction = ''
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        img_h, img_w, img_c = frame.shape
        face_3d = []
        face_2d = []

        # frame.flags.writable = False
        result = face_mesh.process(frame)
        # frame.flags.writable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if result.multi_face_landmarks:
            for face_landmark in result.multi_face_landmarks:
                for idx, lm in enumerate(face_landmark.landmark):
                    if idx in [33, 263, 1, 61, 291, 199]:
                        x = int(lm.x * img_w)
                        y = int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            focal_lenght = 1*img_w
            cam_matrix = np.array(
                [[focal_lenght, 0, img_h / 2], [0, focal_lenght, img_w / 2], [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            horizontal_angle = angles[1] * 360
            vertical_angle = angles[0] * 360 - 13
            print(vertical_angle)
            type = ""
            value = 0
            signal = ''
            last_direction = direction
            if abs(horizontal_angle) > abs(vertical_angle):
                type = "horizontal_angle"
                value = horizontal_angle
            else:
                type = "vertical_angle"
                value = vertical_angle

            if value > 10:
                count += 1
                if type == "horizontal_angle":
                    direction = 'RIGHT'
                else:
                    direction = 'UP'

            elif value < -10:
                count += 1
                if type == "horizontal_angle":
                    direction = 'LEFT'
                else:
                    direction = 'DOWN'
            else:
                count = 0  # reset - looking forward
                direction = ''
            if last_direction != direction and direction != '':  # in case if changed direction - reset
                count = 0

            if count == 2:
                signal = direction
                
            if signal != '':
                ws.send(signal.upper())
    # można by teoretycznie zastanowić się nad sterowaniem za pomocą zmiany kąta, pierwszej pochodnej. Ale nie wiem, czy gra jest warta zachodu. To działa, jako tako, ale działa.
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    # while True:
    #     user_input = input("Enter direction (UP/DOWN/LEFT/RIGHT): ")
    #     if user_input.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]:
    #         ws.send(user_input.upper())
    #     else:
    #         print("Invalid input. Please enter UP, DOWN, LEFT, or RIGHT.")


def on_message(ws, message):
    print(message)


def on_error(ws, error):
    print(error)


def on_close(ws):
    print("Closing")


def on_open(ws):
    # Start a separate thread for user input
    threading.Thread(target=get_user_input, args=(ws,)).start()


# websocket.enableTrace(True)
# ws = websocket.WebSocketApp("ws://localhost:8080/Auth",
#                             on_message=on_message,
#                             on_error=on_error,
#                             on_close=on_close)
ws = websocket.WebSocketApp("ws://192.168.43.187:2137/Auth",
                            on_message=on_message,
                            on_error=on_error,
                            on_close=on_close)
ws.on_open = on_open
ws.run_forever()
