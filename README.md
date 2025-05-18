# SignSpeaks

A real-time sign language recognition system that converts sign language gestures into text using computer vision and machine learning.

## Features

- Real-time sign language recognition using webcam
- Converts sign language gestures to text
- Interactive web interface
- Learning area for sign language practice
- Quiz system for testing knowledge

## Technologies Used

- Python (Primary Programming Language)
- Flask (Web Framework & Backend)
- OpenCV (Computer Vision)
- MediaPipe (Hand Tracking)
- Scikit-learn (Machine Learning)
- HTML/CSS/JavaScript (Frontend)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ayushchauhan22/SignSpeaks.git
cd SignSpeaks
```

2. Install the required dependencies:
```bash
pip install -r libraries.txt
```

3. Run the application:
```bash
python web_app.py
```

4. Open your browser and navigate to `http://localhost:5001` (the server is set to run on port `5001` by default)

## Project File Structure

- `web_app.py`: Main Flask application to run the server
- `sign_language.py`: Sign language recognition logic
- `templates/`: HTML templates
- `assets/`: Static files (Images and icons)
- `models/`: Prediction model files
- `training_data/`: Training data for the model

## Requirements

See `libraries.txt` for the complete list of dependencies.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

Ayush Chauhan, Ayush Gupta, Ansh Sharma, Brahmjot Singh
