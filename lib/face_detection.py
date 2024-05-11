import cv2

def load_face_detector():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

def detect_faces(face_cascade, gray_frame):
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    return faces
