import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import pickle

# Load models and classes
def load_labels(file_path):
    with open(file_path, 'r') as json_file:
        labels = json.load(json_file)
    return {int(k): v for k, v in labels.items()}

# Add error handling for model and camera loading
try:
    alph_labels = load_labels('./classes/ALPH_CLASSES.json')
    num_labels = load_labels('./classes/NUM_CLASSES.json')

    alph_model_dict = pickle.load(open('./models/alph_model.p', 'rb'))
    num_model_dict = pickle.load(open('./models/num_model.p', 'rb'))
except Exception as e:
    st.error(f"Error loading models: {e}")
    alph_labels, num_labels = {}, {}
    alph_model_dict, num_model_dict = {}, {}

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def main():
    st.title("Hand Sign Recognition")

    # Model selection
    model_type = st.selectbox("Select Model", ["Alphabet", "Number"])

    # Load appropriate model
    try:
        if model_type.lower() == "alphabet":
            current_model = alph_model_dict['model']
            current_labels = alph_labels
        else:
            current_model = num_model_dict['model']
            current_labels = num_labels
    except Exception as e:
        st.error(f"Error selecting model: {e}")
        return

    # Image input from user or camera input
    image_source = st.radio("Select Input", ("Upload Image", "Take a Photo"))

    if image_source == "Upload Image":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

        if uploaded_image is not None:
            image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
            frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
            process_image(frame, current_model, current_labels)
    
    elif image_source == "Take a Photo":
        # Camera input for capturing an image
        img = st.camera_input("Click to Take a Photo")

        if img is not None:
            # Read the image from the camera input as a byte object
            image_data = img.getvalue()
            frame = np.frombuffer(image_data, dtype=np.uint8)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            process_image(frame, current_model, current_labels)

def process_image(frame, current_model, current_labels):
    # Process the image
    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hands detection setup
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw hand landmarks
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # Extract hand landmark coordinates
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)

        # Normalize coordinates
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        # Bounding box
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Prediction
        prediction = current_model.predict([np.asarray(data_aux)])
        predicted_character = current_labels[int(prediction[0])]

        # Draw prediction
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,50,0), 3, cv2.LINE_AA)

    # Convert frame to RGB for Streamlit display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, caption="Processed Image", use_container_width=True)

if __name__ == "__main__":
    main()
