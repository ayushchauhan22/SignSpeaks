from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import cv2
import mediapipe as mp
import numpy as np
import pickle
import base64
import io
from PIL import Image
import logging
import traceback
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app with explicit static and template folders
app = Flask(__name__, 
    static_folder='assets',
    static_url_path='/assets',
    template_folder='templates')

# Load the model and scaler
try:
    model_path = os.environ.get('MODEL_PATH', 'models/model.p')
    logger.info(f"Attempting to load model from: {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at: {model_path}")
        raise FileNotFoundError(f"Model file not found at: {model_path}")
        
    model_dict = pickle.load(open(model_path, 'rb'))
    model = model_dict['model']
    scaler = model_dict['scaler']
    labels_dict = model_dict.get('labels_dict', {})
    inverse_labels_dict = {v: k for k, v in labels_dict.items()}
    logger.info("Model and scaler loaded successfully")
    logger.info(f"Available labels: {list(labels_dict.keys())}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Global variables for sentence and prediction tracking
sentence = []
stable_prediction_count = 0
last_prediction = None
STABILITY_THRESHOLD = 6
last_added_sign = None
sign_cooldown = 0
SIGN_COOLDOWN_THRESHOLD = 2
last_sign_time = 0
SIGN_REPEAT_DELAY = 0.7

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sign-language')
def sign_language():
    return render_template('sign_language.html')

@app.route('/learning-area')
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

@app.route('/camera')
def camera():
    try:
        return render_template('sign_language.html')
    except Exception as e:
        logger.error(f"Error rendering camera page: {str(e)}")
        logger.error(traceback.format_exc())
        return "Error loading camera page", 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    try:
        return send_from_directory(app.static_folder, filename)
    except Exception as e:
        logger.error(f"Error serving static file {filename}: {str(e)}")
        logger.error(traceback.format_exc())
        return "File not found", 404

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    try:
        return send_from_directory('assets', filename)
    except Exception as e:
        logger.error(f"Error serving asset file {filename}: {str(e)}")
        logger.error(traceback.format_exc())
        return "File not found", 404

def extract_features(hand_landmarks):
    try:
        data_aux = []
        x_coords = []
        y_coords = []
        
        for landmark in hand_landmarks.landmark:
            x_coords.append(landmark.x)
            y_coords.append(landmark.y)
        
        # Normalize coordinates
        palm_center_x = np.mean(x_coords)
        palm_center_y = np.mean(y_coords)
        
        for landmark in hand_landmarks.landmark:
            x = landmark.x - palm_center_x
            y = landmark.y - palm_center_y
            data_aux.extend([x, y])
        
        # Add hand size and finger angles
        thumb_x = hand_landmarks.landmark[4].x - palm_center_x
        thumb_y = hand_landmarks.landmark[4].y - palm_center_y
        pinky_x = hand_landmarks.landmark[20].x - palm_center_x
        pinky_y = hand_landmarks.landmark[20].y - palm_center_y
        hand_size = np.sqrt((thumb_x - pinky_x)**2 + (thumb_y - pinky_y)**2)
        data_aux.append(hand_size)
        
        for finger_tip in [8, 12, 16, 20]:
            tip_x = hand_landmarks.landmark[finger_tip].x - palm_center_x
            tip_y = hand_landmarks.landmark[finger_tip].y - palm_center_y
            angle = np.arctan2(tip_y, tip_x)
            data_aux.append(angle)
        
        return data_aux
    except Exception as e:
        logger.error(f"Error in extract_features: {str(e)}")
        logger.error(traceback.format_exc())
        return None

@app.route('/sign-language/process_frame', methods=['POST'])
def process_frame():
    global sentence, stable_prediction_count, last_prediction, last_added_sign, sign_cooldown, last_sign_time
    
    try:
        # Get the frame from the request
        if not request.is_json:
            logger.error("Request is not JSON")
            return jsonify({'error': 'Invalid request format'}), 400
            
        if 'frame' not in request.json:
            logger.error("No frame data in request")
            return jsonify({'error': 'No frame data provided'}), 400
            
        frame_data = request.json['frame']
        if not frame_data or not isinstance(frame_data, str):
            logger.error("Invalid frame data format")
            return jsonify({'error': 'Invalid frame data format'}), 400
            
        try:
            # Split the data URL and get the base64 part
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]
            
            frame_bytes = base64.b64decode(frame_data)
            if len(frame_bytes) < 100:  # Basic validation
                logger.error("Frame data too small")
                return jsonify({'error': 'Invalid frame data'}), 400
                
            frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Error decoding frame: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': 'Failed to decode frame'}), 400
        
        if frame is None:
            logger.error("Decoded frame is None")
            return jsonify({'error': 'Failed to decode frame'}), 400
            
        if frame.size == 0:
            logger.error("Decoded frame is empty")
            return jsonify({'error': 'Empty frame'}), 400
        
        # Process the frame
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
        except Exception as e:
            logger.error(f"Error processing frame with MediaPipe: {str(e)}")
            return jsonify({'error': 'Failed to process frame'}), 500
        
        current_time = time.time()
        can_repeat = (current_time - last_sign_time) > SIGN_REPEAT_DELAY
        
        prediction_data = {
            'current_sign': '',
            'top_predictions': [],
            'sentence': ''.join(sentence),
            'landmarks': [],
            'can_repeat': can_repeat
        }
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                try:
                    # Draw landmarks on the frame
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Extract features and make prediction
                    data_aux = extract_features(hand_landmarks)
                    if data_aux is None:
                        continue
                        
                    data_aux_scaled = scaler.transform([data_aux])
                    
                    # Get prediction probabilities
                    prediction_probs = model.predict_proba(data_aux_scaled)[0]
                    top_3_indices = np.argsort(prediction_probs)[-3:][::-1]
                    
                    # Get the prediction
                    prediction = model.predict(data_aux_scaled)[0]
                    
                    # Get the predicted sign
                    if isinstance(prediction, (int, np.integer)):
                        predicted_sign = inverse_labels_dict.get(prediction, 'Unknown')
                    else:
                        predicted_sign = prediction
                    
                    # Update prediction data
                    prediction_data['current_sign'] = predicted_sign
                    prediction_data['top_predictions'] = [
                        {
                            'sign': inverse_labels_dict.get(idx, 'Unknown') if isinstance(idx, (int, np.integer)) else idx,
                            'probability': float(prediction_probs[idx])
                        }
                        for idx in top_3_indices
                    ]
                    
                    # Handle stable predictions
                    if prediction == last_prediction:
                        stable_prediction_count += 1
                    else:
                        stable_prediction_count = 0
                        last_prediction = prediction
                    
                    # Update sentence if prediction is stable
                    if stable_prediction_count == STABILITY_THRESHOLD:
                        if predicted_sign.lower() == 'del':
                            if sentence:
                                sentence.pop()
                                last_added_sign = None
                                sign_cooldown = 0
                        elif predicted_sign.lower() == 'space':
                            sentence.append(' ')
                            last_added_sign = None
                            sign_cooldown = 0
                        else:
                            # Handle repeated signs with time-based cooldown
                            if predicted_sign != last_added_sign or can_repeat:
                                sentence.append(predicted_sign)
                                last_added_sign = predicted_sign
                                last_sign_time = current_time
                                sign_cooldown = 0
                        
                        stable_prediction_count = 0
                        prediction_data['sentence'] = ''.join(sentence)
                except Exception as e:
                    logger.error(f"Error processing hand landmarks: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
        
        # Convert frame to base64 for sending back
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            prediction_data['frame'] = f'data:image/jpeg;base64,{frame_base64}'
        except Exception as e:
            logger.error(f"Error encoding frame: {str(e)}")
            prediction_data['frame'] = None
        
        return jsonify(prediction_data)
        
    except Exception as e:
        logger.error(f"Error in process_frame: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/sign-language/clear_sentence', methods=['POST'])
def clear_sentence():
    global sentence, last_added_sign, sign_cooldown, last_sign_time
    sentence = []
    last_added_sign = None
    sign_cooldown = 0
    last_sign_time = 0
    return jsonify({'status': 'success', 'sentence': ''})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=(os.environ.get('FLASK_ENV', 'production') == 'development')
    ) 