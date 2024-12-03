import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import json
import asyncio
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase


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


# Streamlit UI setup
st.title("Live Hand Sign Recognition")

# Select model type
model_type = st.selectbox("Select Model", ["Alphabet", "Number"])

# Load models
alph_labels, num_labels, alph_model_dict, num_model_dict = load_model()

# Select the appropriate model based on user's choice
current_model = alph_model_dict['model'] if model_type == "Alphabet" else num_model_dict['model']
current_labels = alph_labels if model_type == "Alphabet" else num_labels


# Video Processor for Streamlit WebRTC
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        # Initialize Mediapipe hands for landmark detection
        self.hands = mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3)
    
    def recv(self, frame):
        # Convert the frame to RGB for Mediapipe
        frame_rgb = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2RGB)

        # Process the frame using Mediapipe
        results = self.hands.process(frame_rgb)

        # Initialize variables for hand landmarks and predictions
        data_aux = []
        x_ = []
        y_ = []

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS,
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
            H, W, _ = frame_rgb.shape
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Prediction
            prediction = current_model.predict([np.asarray(data_aux)])
            predicted_character = current_labels[int(prediction[0])]

            # Draw the predicted character on the frame
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(frame_rgb, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                        (255, 255, 255), 3, cv2.LINE_AA)

        return frame_rgb


# Async main function to handle Streamlit WebRTC
async def main():
    st.sidebar.title("WebRTC Settings")
    
    # Start video stream using Streamlit WebRTC
    webrtc_streamer(key="hand-sign-recognition", mode=WebRtcMode.SENDRECV,
                    video_processor_factory=VideoProcessor, async_processing=True)


# Ensure proper event loop handling
if __name__ == "__main__":
    asyncio.run(main())
