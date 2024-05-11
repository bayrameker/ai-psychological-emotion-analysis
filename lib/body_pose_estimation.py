import cv2
import numpy as np

def load_pose_model():
    # Bu fonksiyon, modelinizi yükleyecek
    # Örneğin, OpenPose modelini yüklemek için gereken kodlar buraya yazılabilir
    model = cv2.dnn.readNetFromTensorflow("models/graph_opt.pb")
    return model

def estimate_pose(model, image):
    # Bu fonksiyon, verilen görüntü üzerinde vücut duruşunu tahmin edecek
    frameWidth, frameHeight = image.shape[1], image.shape[0]
    model.setInput(cv2.dnn.blobFromImage(image, 1.0, (frameWidth, frameHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    keypoints = model.forward()
    return keypoints


def draw_keypoints(frame, keypoints, threshold=0.2):
    POSE_PAIRS = [
        [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9],
        [9, 10], [1, 11], [11, 12], [12, 13], [0, 1], [0, 14], [14, 16],
        [0, 15], [15, 17]
    ]
    for pair in POSE_PAIRS:
        partA, partB = pair[0], pair[1]
        if partA < len(keypoints) and partB < len(keypoints) and keypoints[partA] is not None and keypoints[partB] is not None:
            cv2.line(frame, keypoints[partA], keypoints[partB], (0, 255, 0), 2)

    for keypoint in keypoints:
        if keypoint is not None and len(keypoint) == 2:  # Her keypoint için uzunluk kontrolü
            cv2.circle(frame, keypoint, 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)



def analyze_pose(keypoints):
    # keypoints listesinin gerekli uzunlukta olup olmadığını kontrol edin.
    if len(keypoints) > 4:  # En azından 5 keypoints (0'dan 4'e kadar) olduğundan emin olun.
        right_arm_raised = keypoints[4][1] < keypoints[2][1]
        left_arm_raised = keypoints[7][1] < keypoints[5][1] if len(keypoints) > 7 else False

        if right_arm_raised and left_arm_raised:
            return "Her iki el kaldırılmış"
        elif right_arm_raised:
            return "Sağ el kaldırılmış"
        elif left_arm_raised:
            return "Sol el kaldırılmış"
        else:
            return "Normal duruş"
    else:
        return "Yeterli keypoints bulunamadı"



