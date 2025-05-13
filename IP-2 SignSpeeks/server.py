from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import numpy as np
import pickle
import mediapipe as mp
import cv2
import base64
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='.', template_folder='.')
CORS(app)  # This enables CORS for all routes

# Global variables for sentence handling
sentence = []
stable_prediction_count = 0
last_prediction = None
STABILITY_THRESHOLD = 6
last_added_sign = None
sign_cooldown = 0
SIGN_COOLDOWN_THRESHOLD = 2
last_sign_time = 0
SIGN_REPEAT_DELAY = 0.7

# Serve HTML files
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/learningArea')
def learning_area():
    return render_template('learningArea.html')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/quiz')
def quiz():
    return render_template('quiz.html')

@app.route('/camera-quiz')
def camera_quiz():
    return render_template('camera-quiz.html')

# Serve static files
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

# Load the trained model and metadata
print("Loading model and metadata...")
try:
    with open('model.p', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        scaler = model_data['scaler']
        labels_dict = model_data['labels_dict']
        valid_classes = model_data['valid_classes']
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe Hands with more sensitive parameters
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_features(landmarks):
    """Extract features from hand landmarks."""
    try:
        data_aux = []
        x_coords = []
        y_coords = []
        
        for landmark in landmarks.landmark:
            x_coords.append(landmark.x)
            y_coords.append(landmark.y)
        
        # Normalize coordinates
        palm_center_x = np.mean(x_coords)
        palm_center_y = np.mean(y_coords)
        
        for landmark in landmarks.landmark:
            x = landmark.x - palm_center_x
            y = landmark.y - palm_center_y
            data_aux.extend([x, y])
        
        return np.array(data_aux)
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        return None

def process_frame(frame_data):
    """Process a frame and detect hand landmarks."""
    global sentence, stable_prediction_count, last_prediction, last_added_sign, sign_cooldown, last_sign_time
    
    try:
        # Decode base64 image
        encoded_data = frame_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = hands.process(frame_rgb)
        
        # Prepare debug image
        debug_image = frame_rgb.copy()
        
        if results.multi_hand_landmarks:
            # Draw hand landmarks on debug image
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    debug_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Get landmarks for the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract features
            features = extract_features(hand_landmarks)
            if features is not None:
                # Scale features
                features_scaled = scaler.transform([features])
                
                # Get prediction probabilities
                prediction_probs = model.predict_proba(features_scaled)[0]
                prediction = model.predict(features_scaled)[0]
                
                # Get top 3 predictions
                top_3_indices = np.argsort(prediction_probs)[-3:][::-1]
                top_predictions = [
                    {
                        'sign': labels_dict.get(idx, 'Unknown'),
                        'probability': float(prediction_probs[idx])
                    }
                    for idx in top_3_indices
                ]
                
                # Handle prediction stability
                predicted_sign = labels_dict.get(prediction, 'Unknown')
                if prediction == last_prediction:
                    stable_prediction_count += 1
                else:
                    stable_prediction_count = 0
                    last_prediction = prediction
                
                # Update sentence if prediction is stable
                if stable_prediction_count >= STABILITY_THRESHOLD:
                    if predicted_sign.lower() == 'del':
                        if sentence:
                            sentence.pop()
                    elif predicted_sign.lower() == 'space':
                        sentence.append(' ')
                    else:
                        sentence.append(predicted_sign)
                    stable_prediction_count = 0
                
                # Convert debug image to base64
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
                debug_image_base64 = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')
                
                return {
                    'status': 'success',
                    'current_sign': predicted_sign,
                    'top_predictions': top_predictions,
                    'sentence': ''.join(sentence),
                    'debug_image': debug_image_base64
                }
        
        # If no hand detected
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
        debug_image_base64 = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')
        
        return {
            'status': 'no_hand_detected',
            'current_sign': '',
            'top_predictions': [],
            'sentence': ''.join(sentence),
            'debug_image': debug_image_base64
        }
            
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return {
            'status': 'error',
            'error': str(e),
            'current_sign': '',
            'top_predictions': [],
            'sentence': ''.join(sentence),
            'debug_image': None
        }

@app.route('/sign-language/process_frame', methods=['POST'])
def process_frame_endpoint():
    """Handle frame processing requests."""
    try:
        data = request.json
        frame_data = data.get('frame')
        
        if not frame_data:
            return jsonify({'error': 'No frame data provided'}), 400
        
        result = process_frame(frame_data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in process_frame endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/sign-language/clear_sentence', methods=['POST'])
def clear_sentence():
    """Clear the current sentence."""
    global sentence, last_added_sign, sign_cooldown, last_sign_time
    sentence = []
    last_added_sign = None
    sign_cooldown = 0
    last_sign_time = 0
    return jsonify({'status': 'success', 'sentence': ''})

if __name__ == '__main__':
    print("Starting SignSpeaks server...")
    app.run(host='0.0.0.0', port=5001, debug=True) 