import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

def load_emotion_model():
    model = load_model('models/FER_model.h5')
    return model

def predict_emotion(model, face_image):
    face_image = cv2.resize(face_image, (48, 48))
    face_image = face_image.astype('float')/255.0
    face_image = img_to_array(face_image)
    face_image = np.expand_dims(face_image, axis=0)
    predictions = model.predict(face_image)[0]
    return np.argmax(predictions)
