import cv2
import numpy as np
import av
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.title('AppliedRoots Project')


cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
fingerCoordinates = [(8, 6), (12, 10), (16, 14), (20, 18)]
thumbCoordinate = (4,2)


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    multiLandMarks = results.multi_hand_landmarks

    if multiLandMarks:
        handPoints = []
        for handLms in multiLandMarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            for idx, lm in enumerate(handLms.landmark):
                # print(idx,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                handPoints.append((cx, cy))

        for point in handPoints:
            cv2.circle(img, point, 10, (0, 0, 255), cv2.FILLED)

        upCount = 0
        for coordinate in fingerCoordinates:
            if handPoints[coordinate[0]][1] < handPoints[coordinate[1]][1]:
                upCount += 1
        if handPoints[thumbCoordinate[0]][0] > handPoints[thumbCoordinate[1]][0]:
            upCount += 1

        cv2.putText(img, str(upCount), (150,150), cv2.FONT_HERSHEY_PLAIN, 12, (0,255,0), 12)

    cv2.imshow("Project Applied Root", img)
    cv2.waitKey(1)
cv2.destroyWindow('Project Applied Root')
# exit()

# if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# # cap = cv2.VideoCapture(0)

# # mpHands = mp.solutions.hands
# # hands = mpHands.Hands()
# # mpDraw = mp.solutions.drawing_utils
# fingerCoordinates = [(8, 6), (12, 10), (16, 14), (20, 18)]
# thumbCoordinate = (4,2)

# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     model_complexity=0,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# def process(image):
#     # Draw the hand annotations on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     if results.multi_hand_landmarks:
#       for hand_landmarks in results.multi_hand_landmarks:
#         mp_drawing.draw_landmarks(
#             image,
#             hand_landmarks,
#             mp_hands.HAND_CONNECTIONS,
#             mp_drawing_styles.get_default_hand_landmarks_style(),
#             mp_drawing_styles.get_default_hand_connections_style())
      
    
#     return cv2.flip(image, 1)



# RTC_CONFIGURATION = RTCConfiguration(
#     {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
# )

# class VideoProcessor:
#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")

#         img = process(img)

#         return av.VideoFrame.from_ndarray(img, format="bgr24")

# webrtc_ctx = webrtc_streamer(
#     key="WYH",
#     mode=WebRtcMode.SENDRECV,
#     rtc_configuration=RTC_CONFIGURATION,
#     media_stream_constraints={"video": True, "audio": False},
#     video_processor_factory=VideoProcessor,
#     async_processing=True,
# )
