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
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

def process_hand_sign(image_data, model, labels):
    # Convert base64 to OpenCV image
    import base64
    import io
    from PIL import Image
    
    # Decode base64 image
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes))
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Process the frame
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
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels[int(prediction[0])]
        
        return predicted_character
    
    return "No hand detected"

def main():
    st.title("Hand Sign Recognition")

    # Model selection
    model_type = st.selectbox("Select Model", ["Alphabet", "Number"])

    # JavaScript for camera access
    camera_html = """
    <div id="camera-container" style="width:100%; max-width:640px;">
        <video id="video" style="width:100%;" autoplay playsinline></video>
        <canvas id="canvas" style="display:none;"></canvas>
    </div>
    <script>
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const constraints = { video: { facingMode: "user" } };

    async function startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = stream;
        } catch (err) {
            console.error("Error accessing camera:", err);
            Streamlit.setComponentValue("Camera access denied");
        }
    }

    function captureImage() {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg');
        Streamlit.setComponentValue(imageData);
    }

    startCamera();

    // Set up button to capture image
    Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, () => {
        const captureButton = document.createElement('button');
        captureButton.textContent = 'Capture';
        captureButton.onclick = captureImage;
        document.getElementById('camera-container').appendChild(captureButton);
    });
    </script>
    """
    
    # Render camera HTML
    image_data = st.components.v1.html(camera_html, height=500)

    # Process captured image
    if image_data:
        try:
            # Select appropriate model based on user choice
            if model_type.lower() == "alphabet":
                current_model = alph_model_dict['model']
                current_labels = alph_labels
            else:
                current_model = num_model_dict['model']
                current_labels = num_labels

            # Process the image
            result = process_hand_sign(image_data, current_model, current_labels)
            st.write(f"Detected Sign: {result}")
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
