from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import json
import pickle

app = Flask(__name__)

# Global variables for model and labels
current_model = None
current_labels = None
current_model_type = None

# Load models and classes
with open('./classes/ALPH_CLASSES.json', 'r') as json_file:
    alph_labels = json.load(json_file)
alph_labels = {int(k): v for k, v in alph_labels.items()}

with open('./classes/NUM_CLASSES.json', 'r') as json_file:
    num_labels = json.load(json_file)
num_labels = {int(k): v for k, v in num_labels.items()}

alph_model_dict = pickle.load(open('./models/alph_model.p', 'rb'))
num_model_dict = pickle.load(open('./models/num_model.p', 'rb'))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

@app.route('/load_model/<model_type>')
def load_model(model_type):
    global current_model, current_labels, current_model_type
    if model_type == 'alphabet':
        current_model = alph_model_dict['model']
        current_labels = alph_labels
        current_model_type = 'alphabet'
    elif model_type == 'number':
        current_model = num_model_dict['model']
        current_labels = num_labels
        current_model_type = 'number'
    else:
        return jsonify(success=False)
    return jsonify(success=True)

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks and current_model is not None:
            hand_landmarks = results.multi_hand_landmarks[0]

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = current_model.predict([np.asarray(data_aux)])
            predicted_character = current_labels[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)


        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)