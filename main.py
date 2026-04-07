"""
Author: Amr Elsersy
Description: Live Camera Demo using opencv dnn face detection & Emotion Recognition
Updated: Added emotion stability interaction (20 frames) + voice assistant
"""

import sys
import time
import argparse
import cv2
import numpy as np
import torch
import torchvision.transforms.transforms as transforms
import pyttsx3

from face_detector.face_detector import DnnDetector, HaarCascadeDetector
from model.model import Mini_Xception
from utils import get_label_emotion, histogram_equalization
from face_alignment.face_alignment import FaceAlignment

sys.path.insert(1, 'face_detector')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Voice Engine ---------------- #
engine = pyttsx3.init()
engine.setProperty('rate', 160)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# -------------- Interaction Logic ------------- #
def interact(emotion):
    emotion = emotion.lower()

    responses = {
        "sad": "You seem sad. Do you want to talk about it?",
        "happy": "You look happy! Shall I play your favorite music?",
        "angry": "You seem angry. Let's take a deep breath together.",
        "surprise": "Oh! That surprised you! What happened?",
        "fear": "You look worried. Is everything okay?",
        "disgust": "Something bothering you?",
        "neutral": "You look calm. How can I assist you?"
    }

    message = responses.get(emotion, "")
    if message:
        print("Assistant:", message)
        speak(message)
        return message
    return ""

# -------------------- MAIN -------------------- #
def main(args):

    # Load model
    mini_xception = Mini_Xception().to(device)
    mini_xception.eval()
    checkpoint = torch.load(args.pretrained, map_location=device)
    mini_xception.load_state_dict(checkpoint['mini_xception'])

    face_alignment = FaceAlignment()

    # Face detector
    root = 'face_detector'
    if args.haar:
        face_detector = HaarCascadeDetector(root)
    else:
        face_detector = DnnDetector(root)

    # Video source
    if args.path:
        video = cv2.VideoCapture(args.path)
    else:
        video = cv2.VideoCapture(0)

    print("video.isOpened:", video.isOpened())

    # -------- Emotion Stability Variables -------- #
    stable_emotion = None
    emotion_counter = 0
    triggered = False
    STABLE_THRESHOLD = 5
    cooldown_time = 10  # seconds
    last_trigger_time = 0
    interaction_text = ""

    t1 = 0

    while video.isOpened():

        ret, frame = video.read()
        if not ret:
            break

        if args.path:
            frame = cv2.resize(frame, (640, 480))

        # FPS calculation
        t2 = time.time()
        fps = round(1 / (t2 - t1 + 1e-5))
        t1 = t2

        faces = face_detector.detect_faces(frame)

        for face in faces:
            (x, y, w, h) = face

            input_face = face_alignment.frontalize_face(face, frame)
            input_face = cv2.resize(input_face, (48, 48))
            input_face = histogram_equalization(input_face)

            input_face = transforms.ToTensor()(input_face).to(device)
            input_face = torch.unsqueeze(input_face, 0)

            with torch.no_grad():
                emotion_pred = mini_xception(input_face)
                emotion_index = torch.argmax(emotion_pred).item()
                emotion = get_label_emotion(emotion_index)

            # -------- Emotion Stability Logic -------- #
            if emotion == stable_emotion:
                emotion_counter += 1
            else:
                stable_emotion = emotion
                emotion_counter = 1
                triggered = False

            # Trigger after stable frames + cooldown
            if (emotion_counter >= STABLE_THRESHOLD and
                not triggered and
                time.time() - last_trigger_time > cooldown_time):

                interaction_text = interact(stable_emotion)
                triggered = True
                last_trigger_time = time.time()

            # Draw face box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Emotion label
            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Show interaction text
        if interaction_text:
            cv2.putText(frame, interaction_text,
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2)

        # FPS display
        cv2.putText(frame, f"FPS: {fps}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

        cv2.imshow("Emotion Recognition Assistant", frame)

        if cv2.waitKey(1) & 0xff == 27:
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--haar', action='store_true')
    parser.add_argument('--pretrained', type=str,
                        default='checkpoint/model_weights/weights_epoch_75.pth.tar')
    parser.add_argument('--path', type=str, default='')
    args = parser.parse_args()

    main(args)
