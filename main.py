import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from collections import defaultdict, Counter

from lib.face_detection import load_face_detector, detect_faces
from lib.emotion_recognition import load_emotion_model, predict_emotion

def main():
    cap = cv2.VideoCapture('data/video/25snemotion.mp4')
    face_cascade = load_face_detector()
    emotion_model = load_emotion_model()
    font = ImageFont.truetype("arial.ttf", 24)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.namedWindow('Emotion Detector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Emotion Detector', width, height)

    diagnosis_interval = 10
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_per_interval = int(fps * diagnosis_interval)
    emotions_dict = defaultdict(list)
    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(face_cascade, gray)

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        for (x, y, w, h) in faces:
            face_image = gray[y:y+h, x:x+w]
            emotion_label = predict_emotion(emotion_model, face_image)
            label_map = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']
            predicted_emotion = label_map[emotion_label]
            draw.text((x, max(0, y - 10)), predicted_emotion, font=font, fill=(0, 255, 0))

            interval = current_frame // frames_per_interval
            emotions_dict[interval].append(predicted_emotion)

        frame_with_text = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        cv2.imshow('Emotion Detector', frame_with_text)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        current_frame += 1

    cap.release()
    cv2.destroyAllWindows()

    # Analyze and present emotion analysis results
    for interval, emotions in emotions_dict.items():
        emotion_count = Counter(emotions)
        print(f"Time Interval {interval * diagnosis_interval} - {(interval + 1) * diagnosis_interval} seconds:")
        print(emotion_count)
        plt.figure(figsize=(10, 4))
        plt.bar(emotion_count.keys(), emotion_count.values())
        plt.title(f"Emotion Distribution from {interval * diagnosis_interval} to {(interval + 1) * diagnosis_interval} seconds")
        plt.xlabel('Emotions')
        plt.ylabel('Count')
        plt.show()

if __name__ == '__main__':
    main()
