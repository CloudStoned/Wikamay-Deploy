import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import json

# Load the labels and models
def load_labels(file_path):
    with open(file_path, 'r') as json_file:
        labels = json.load(json_file)
    return {int(k): v for k, v in labels.items()}

# Model loading
def load_model():
    try:
        alph_labels = load_labels('./classes/ALPH_CLASSES.json')
        num_labels = load_labels('./classes/NUM_CLASSES.json')
        alph_model_dict = pickle.load(open('./models/alph_model.p', 'rb'))
        num_model_dict = pickle.load(open('./models/num_model.p', 'rb'))
        return alph_labels, num_labels, alph_model_dict, num_model_dict
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

# Mediapipe setup for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the models and labels
alph_labels, num_labels, alph_model_dict, num_model_dict = load_model()

# Streamlit UI setup
st.title("Live Hand Sign Recognition")

# Select model type
model_type = st.selectbox("Select Model", ["Alphabet", "Number"])

# Select the appropriate model based on user's choice
current_model = alph_model_dict['model'] if model_type == "Alphabet" else num_model_dict['model']
current_labels = alph_labels if model_type == "Alphabet" else num_labels

# Start video stream
run = st.checkbox("Run Camera Stream")
FRAME_WINDOW = st.camera_input("Camera")

if run:
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        st.error("Unable to access camera.")
    else:
        # Initialize mediapipe hands for landmark detection
        hands = mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3)

        while run:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to capture frame.")
                break

            # Convert the frame to RGB for Mediapipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame using Mediapipe
            results = hands.process(frame_rgb)

            # Initialize variables for hand landmarks and predictions
            data_aux = []
            x_ = []
            y_ = []

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())

                # Extract landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalize and prepare data for prediction
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Bounding box for the hand
                H, W, _ = frame.shape
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Prediction
                prediction = current_model.predict([np.asarray(data_aux)])
                predicted_character = current_labels[int(prediction[0])]

                # Draw the predicted character on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                            (255, 255, 255), 3, cv2.LINE_AA)

            # Convert the frame to RGB and display it
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)

        # Release the camera when done
        camera.release()
