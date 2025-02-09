
import streamlit as st
import cv2
import numpy as np
import dlib
from imutils import face_utils
import time
import pygame
import threading
from twilio.rest import Client
import os

# Initialize pygame mixer for alert sound
pygame.mixer.init()

# Flag to check if sound is already playing
sound_played = False

TWILIO_ACCOUNT_SID = "ACc66199ae780625109cd88436500c48ef"
TWILIO_AUTH_TOKEN = "ea9b6e4d57293534def3557fc3bb5b15"
TWILIO_PHONE_NUMBER = "+12182506379"
EMERGENCY_CONTACT = "+917218824923"

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


# Function to play the sound in a separate thread
def play_alert(sound_file):
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play(loops=-1)


# Function to play sound non-blocking
def play_sound_in_thread():
    global sound_played
    if not sound_played:
        sound_thread = threading.Thread(target=play_alert, args=("alert_beep.mp3",))
        sound_thread.start()
        sound_played = True


# Function to make an emergency call using Twilio
emergency_called = False


# Function to make emergency call twice
def make_emergency_call():
    global emergency_called
    if emergency_called:
        return  # Prevent multiple calls

    emergency_called = True  # Set flag to True after first call
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    # First Call
    call = client.calls.create(
        to=EMERGENCY_CONTACT,
        from_=TWILIO_PHONE_NUMBER,
        twiml='<Response><Say>This is an emergency alert. The driver is drowsy. Please check immediately.</Say></Response>'
    )


# Dashboard Layout
st.title("🚗 Drowsy Driver Detection with Emergency Call 🚨")
st.sidebar.header("Settings")

# Sidebar Controls
eye_closure_threshold = st.sidebar.slider("Eye Closure Threshold", 0.2, 0.5, 0.25, 0.01)
yawn_threshold = st.sidebar.slider("Yawning Threshold (MAR)", 20, 60, 35, 1)
head_pose_threshold = st.sidebar.slider("Head Pose Threshold (Yaw)", 15, 45, 20, 1)
roll_pose_threshold = st.sidebar.slider("Roll Pose Threshold (Tilt)", 10, 30, 15, 1)
sleep_threshold = st.sidebar.slider("Sleeping Detection Threshold (seconds)", 1, 5, 2, 1)

# Button to start/stop the detection
start_detection = st.sidebar.button("Start Detection")

# Load pre-trained models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Function to compute Euclidean distance
def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)


# Function to calculate Eye Aspect Ratio (EAR)
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    return up / (2.0 * down)


# Function to calculate Mouth Aspect Ratio (MAR) for yawning detection
def calculate_mar(landmarks):
    top_lip = compute(landmarks[50], landmarks[58])
    bottom_lip = compute(landmarks[52], landmarks[56])
    horizontal = compute(landmarks[48], landmarks[54])
    return (top_lip + bottom_lip) / (2.0 * horizontal)


# Function to estimate head pose
def detect_head_pose(landmarks, frame_shape):
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right Mouth corner
    ], dtype=np.float32)

    image_points = np.array([
        landmarks[30],  # Nose tip
        landmarks[8],  # Chin
        landmarks[36],  # Left eye left corner
        landmarks[45],  # Right eye right corner
        landmarks[48],  # Left Mouth corner
        landmarks[54]  # Right Mouth corner
    ], dtype=np.float32)

    focal_length = frame_shape[1]
    center = (frame_shape[1] / 2, frame_shape[0] / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

    return pose_angles


# Main detection loop
if start_detection:
    cap = cv2.VideoCapture(0)
    status_display = st.empty()
    video_display = st.image([])
    start_sleep_time = None
    yawn_start_time = None
    fps = 5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera not detected. Please check your webcam!")
            break

        frame = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        status = "Active"
        color = (0, 255, 0)

        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            left_eye_ratio = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40],
                                     landmarks[39])
            right_eye_ratio = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46],
                                      landmarks[45])
            eyes_closed = left_eye_ratio < eye_closure_threshold and right_eye_ratio < eye_closure_threshold

            if eyes_closed:
                if start_sleep_time is None:
                    start_sleep_time = time.time()

                elif time.time() - start_sleep_time > sleep_threshold:
                    play_sound_in_thread()

                    status = "Sleeping!"
                    color = (0, 0, 255)
                    if not emergency_called:  # Make sure call happens only once
                        make_emergency_call()
                else:
                    status = "Drowsy!"
                    color = (0, 0, 255)
            else:
                start_sleep_time = None
                status = "Active!"
                color = (0, 255, 0)
                sound_played = False
                pygame.mixer.music.stop()

                emergency_called = False  # RESET when driver is active again

            mar = calculate_mar(landmarks)
            if mar > yawn_threshold:
                status = "Yawning!"
                color = (0, 255, 255)

            pose_angles = detect_head_pose(landmarks, frame.shape)
            if abs(pose_angles[1]) > head_pose_threshold or abs(pose_angles[2]) > roll_pose_threshold:
                status = "Distracted!"
                color = (255, 0, 255)

            status_display.markdown(f"### Status: {status}")
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        video_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        time.sleep(1 / fps)

    cap.release()
