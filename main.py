import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from collections import Counter

# Yüz tanıma için OpenCV'nin hazır modelini yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Duygu tanıma modelini yükle
emotion_model = load_model('FER_model.h5')

# Videoyu yükle
cap = cv2.VideoCapture('emotion.mp4')

# Duyguların kaydedildiği sözlük
emotions_dict = {}

while True:
    # Video'dan bir frame al
    ret, frame = cap.read()
    if not ret:
        break

    # Griye çevir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri tespit et
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Mevcut frame numarası

    for (x, y, w, h) in faces:
        # Yüz bölgesini kes
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float')/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Duygu tahmini yap
        predictions = emotion_model.predict(roi)[0]
        emotion_label = np.argmax(predictions)

        # Duygu etiketini ekrana yaz
        label_map = ['Kızgın', 'Tiksindirici', 'Korkmuş', 'Mutlu', 'Üzgün', 'Şaşırmış', 'Nötr']
        predicted_emotion = label_map[emotion_label]
        cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # Duygu kaydı
        if frame_number not in emotions_dict:
            emotions_dict[frame_number] = []
        emotions_dict[frame_number].append(predicted_emotion)

    # Videonun işlenmiş halini göster
    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()

# Duygu analizi sonuçları
all_emotions = [emotion for frame_emotions in emotions_dict.values() for emotion in frame_emotions]
emotion_counts = Counter(all_emotions)

print("Duygu Sıklığı Analizi:")
print(emotion_counts)

# Duygusal Süreklilik Analizi
most_common_emotion, freq = emotion_counts.most_common(1)[0]
print(f"En sık rastlanan duygu: {most_common_emotion} ({freq} kez)")

# Duygu Sürekliliği
continuity_threshold = 5  # Bu eşik değer altında sürekli duygular
continuous_emotions = {emotion: count for emotion, count in emotion_counts.items() if count > continuity_threshold}
print("Sürekli Gözlenen Duygular:")
print(continuous_emotions)
