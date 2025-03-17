from ultralytics import YOLO
import cv2
import mediapipe as mp
import time

# Load YOLOv8 Pose model
model = YOLO("yolov8n-pose.pt")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=10, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing Utilities
mp_drawing = mp.solutions.drawing_utils

# Initialize camera
cap = cv2.VideoCapture(0)

def detect_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

    threshold = 0.05
    thumb_raised = thumb_tip.y < thumb_ip.y - threshold
    index_raised = index_tip.y < index_pip.y - threshold
    middle_raised = middle_tip.y < middle_pip.y - threshold
    ring_raised = ring_tip.y < ring_pip.y - threshold
    pinky_raised = pinky_tip.y < pinky_pip.y - threshold

    if thumb_raised and index_raised and not middle_raised and not ring_raised and pinky_raised:
        return "Metal"
    elif thumb_raised and index_raised and middle_raised and ring_raised and pinky_raised:
        return "Jari Terbuka"
    elif not thumb_raised and not index_raised and not middle_raised and not ring_raised and not pinky_raised:
        return "Jari Tertutup"
    elif not thumb_raised and index_raised and middle_raised and not ring_raised and not pinky_raised:
        return "Peace"
    elif thumb_raised and not index_raised and not middle_raised and not ring_raised and not pinky_raised:
        return "Jempol"
    elif not thumb_raised and index_raised and not middle_raised and not ring_raised and not pinky_raised:
        return "Telunjuk"
    elif not thumb_raised and not index_raised and middle_raised and not ring_raised and not pinky_raised:
        return "Tengah"
    elif not thumb_raised and not index_raised and not middle_raised and ring_raised and not pinky_raised:
        return "Manis"
    elif not thumb_raised and not index_raised and not middle_raised and not ring_raised and pinky_raised:
        return "Kelingking"
    else:
        return "Unknown"

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_results = hands.process(image_rgb)
    
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = 0, 0
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                x_min = min(x, x_min)
                x_max = max(x, x_max)
                y_min = min(y, y_min)
                y_max = max(y, y_max)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            gesture = detect_gesture(hand_landmarks)
            cv2.putText(frame, f"{gesture}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            print(f"Gesture: {gesture}, Bounding Box: ({x_min}, {y_min}), ({x_max}, {y_max})")
    
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    print(f"FPS: {fps:.2f}")
    
    cv2.imshow("Multi-Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
