# SignSpeaks - Empowering Communication Through ML

SignSpeaks is a web application that translates sign language gestures into text in real-time using computer vision and machine learning.

## Features
- Real-time sign language translation
- Interactive learning area
- Camera-based gesture recognition
- Support for ASL (American Sign Language)

## Setup Instructions

### Prerequisites
- Python 3.10 or higher
- Webcam
- Required Python packages (install using `pip install -r requirements.txt`)

### Installation
1. Clone the repository:
```bash
git clone https://github.com/ayushchauhan22/SignSpeaks.git
cd SignSpeaks
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the model files:
   - Due to size limitations, the model files are not included in the repository
   - Contact the repository owner to obtain the required model files
   - Place the model files in the `IP-2 SignSpeeks` directory

4. Run the application:
```bash
python web_app.py
```

5. Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

## Project Structure
- `web_app.py` - Main Flask application
- `sign_language.py` - Sign language processing module
- `IP-2 SignSpeeks/` - Contains frontend files and model
- `templates/` - HTML templates
- `static/` - Static files (CSS, JavaScript, images)

## Note
The model files (`model.p` and related data files) are not included in this repository due to size limitations. Please contact the repository owner to obtain these files.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
