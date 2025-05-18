from flask import Blueprint, render_template, request, jsonify
import pickle
import cv2
import mediapipe as mp
import numpy as np
import logging
import base64
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

sign_language_bp = Blueprint('sign_language', __name__)

# Load the model and scaler
try:
    model_path = 'IP-2 SignSpeeks/model.p'
    logger.debug(f"Loading model from {model_path}")
    model_dict = pickle.load(open(model_path, 'rb'))
    model = model_dict['model']
    scaler = model_dict['scaler']
    labels_dict = model_dict.get('labels_dict', {})
    inverse_labels_dict = {v: k for k, v in labels_dict.items()}
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
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
STABILITY_THRESHOLD = 10

@sign_language_bp.route('/sign-language')
def sign_language():
    return render_template('sign_language.html')

@sign_language_bp.route('/sign-language/clear_sentence', methods=['POST'])
def clear_sentence():
    global sentence
    sentence = []
    return jsonify({'status': 'success'})

@sign_language_bp.route('/sign-language/process_frame', methods=['POST'])
def process_frame():
    global sentence, stable_prediction_count, last_prediction
    
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({'error': 'No frame data provided'}), 400

        # Decode the base64 image
        frame_data = data['frame'].split(',')[1]
        frame_bytes = base64.b64decode(frame_data)
        frame = Image.open(io.BytesIO(frame_bytes))
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        # Process the frame with Mediapipe
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract features
                features = extract_features(hand_landmarks)
                if features is not None:
                    # Scale features
                    features_scaled = scaler.transform([features])
                    
                    # Make prediction
                    prediction = model.predict(features_scaled)[0]
                    probabilities = model.predict_proba(features_scaled)[0]
                    
                    # Get top 3 predictions
                    top_indices = probabilities.argsort()[-3:][::-1]
                    top_predictions = [
                        {
                            'sign': inverse_labels_dict.get(idx, f'Sign {idx}').replace('Sign ', ''),
                            'probability': float(probabilities[idx])
                        }
                        for idx in top_indices
                    ]
                    
                    # Update sentence based on prediction stability
                    current_sign = inverse_labels_dict.get(prediction, f'Sign {prediction}').replace('Sign ', '')
                    
                    # Handle special signs (space and delete)
                    if current_sign.lower() == 'Space' or current_sign == '_':
                        current_sign = 'SPACE'  # For display purposes
                        if current_sign == last_prediction:
                            stable_prediction_count += 1
                            if stable_prediction_count >= STABILITY_THRESHOLD:
                                sentence.append(' ')
                                stable_prediction_count = 0
                                last_prediction = None  # Reset to allow next character
                        else:
                            stable_prediction_count = 0
                            last_prediction = current_sign
                    elif current_sign.lower() == 'Delete' or current_sign.lower() == 'delete':
                        current_sign = 'DEL'  # For display purposes
                        if current_sign == last_prediction:
                            stable_prediction_count += 1
                            if stable_prediction_count >= STABILITY_THRESHOLD and sentence:  # Check if sentence is not empty
                                sentence.pop()  # Remove last character
                                stable_prediction_count = 0
                        else:
                            stable_prediction_count = 0
                            last_prediction = current_sign
                    else:
                        if current_sign == last_prediction:
                            stable_prediction_count += 1
                            if stable_prediction_count >= STABILITY_THRESHOLD:
                                if not sentence or sentence[-1] != current_sign:
                                    sentence.append(current_sign)
                                stable_prediction_count = 0
                                last_prediction = None  # Reset to allow next character
                        else:
                            stable_prediction_count = 0
                            last_prediction = current_sign

                    return jsonify({
                        'sentence': ''.join(sentence),
                        'current_sign': current_sign,
                        'top_predictions': top_predictions,
                        'has_space': current_sign == 'SPACE',
                        'is_delete': current_sign == 'DEL'
                    })

        return jsonify({
            'sentence': ''.join(sentence),
            'current_sign': 'No sign detected',
            'top_predictions': [],
            'has_space': False,
            'is_delete': False
        })

    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def extract_features(hand_landmarks):
    try:
        # Extract x and y coordinates
        x_coords = []
        y_coords = []
        for landmark in hand_landmarks.landmark:
            x_coords.append(landmark.x)
            y_coords.append(landmark.y)
        
        # Calculate palm center
        palm_center_x = np.mean(x_coords)
        palm_center_y = np.mean(y_coords)
        
        # Calculate relative coordinates and hand size
        features = []
        for landmark in hand_landmarks.landmark:
            # Calculate relative coordinates to palm center
            x = landmark.x - palm_center_x
            y = landmark.y - palm_center_y
            features.extend([x, y])
        
        # Add hand size feature (distance between thumb and pinky)
        thumb_x = hand_landmarks.landmark[4].x - palm_center_x
        thumb_y = hand_landmarks.landmark[4].y - palm_center_y
        pinky_x = hand_landmarks.landmark[20].x - palm_center_x
        pinky_y = hand_landmarks.landmark[20].y - palm_center_y
        hand_size = np.sqrt((thumb_x - pinky_x)**2 + (thumb_y - pinky_y)**2)
        features.append(hand_size)
        
        # Ensure we have exactly 47 features
        if len(features) > 47:
            features = features[:47]
        elif len(features) < 47:
            features.extend([0] * (47 - len(features)))
            
        return features
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}", exc_info=True)
        return None
