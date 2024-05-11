

# Video Emotion and Pose Recognition

This project integrates face detection, emotion recognition, and body pose estimation to analyze videos in real-time. It detects faces in the video stream, identifies emotions, and analyzes the body posture using machine learning models.

## Features

- **Face Detection**: Detects faces in video frames.
- **Emotion Recognition**: Identifies emotions from facial expressions.
- **Pose Estimation**: Analyzes the body posture by detecting key points.

## Requirements

To run this project, you will need the following:
- Python 3.x
- OpenCV
- NumPy
- Pillow
- Matplotlib

## Installation

Follow these steps to set up the project environment:

1. Clone the repository:
   ```
   git clone https://github.com/scriptchief/ai-psychological-emotion-analysis
   cd ai-psychological-emotion-analysis
   ```

2. Install the required packages:
   ```
   pip install opencv-python numpy pillow matplotlib
   ```

## Usage

To run the main application, execute:
```
python main.py
```

The script will start processing the video specified in the code and display the emotion and pose analysis results in real-time.

## Structure

- **main.py**: Contains the main workflow including video capture, processing, and display of results.
- **lib/face_detection.py**: Module for face detection utilities.
- **lib/emotion_recognition.py**: Module for emotion recognition using pre-trained models.
- **lib/body_pose_estimation.py**: Module for body pose estimation.

Each module is responsible for a specific part of the analysis and can be modified independently.

## Customization

You can customize the parameters and models used in:
- `lib/face_detection.py`: Update face detector settings.
- `lib/emotion_recognition.py`: Switch or retrain the emotion recognition model.
- `lib/body_pose_estimation.py`: Change pose estimation models or adjust detection parameters.

## Contributing

Contributions to improve the project are welcome. Please fork the repository and submit a pull request with your changes.

## License

Specify your license or state that the project is open-source.

