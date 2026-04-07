"""
Emotion + Eye Blink + Voice AI Assistant + Serial Signals
FIXED VERSION (Voice + Thread + Stability)
"""

import sys, os, time, threading, argparse
import cv2
import torch
import numpy as np
import torchvision.transforms.transforms as transforms
import speech_recognition as sr
import serial
from groq import Groq
from scipy.spatial import distance
from imutils import face_utils
import dlib
import imutils
import pyttsx3

from face_detector.face_detector import DnnDetector, HaarCascadeDetector
from model.model import Mini_Xception
from utils import get_label_emotion, histogram_equalization
from face_alignment.face_alignment import FaceAlignment

# ---------------- DEVICE ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- GROQ ---------------- #
client = Groq(api_key="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")

# ---------------- SERIAL PORT ---------------- #
try:
    ser = serial.Serial('COM3', 9600, timeout=1)
    time.sleep(2)
except:
    ser = None
    print("Warning: Serial port not available")

def safe_serial_write(signal):
    if ser:
        try:
            ser.write(signal)
        except:
            pass

# ---------------- TTS (FIXED) ---------------- #
def speak(text):
    try:
        engine = pyttsx3.init()   # new instance every time
        engine.setProperty('rate', 170)
        engine.setProperty('volume', 1.0)

        safe_serial_write(b'2')  # speaking start

        engine.say(text)
        engine.runAndWait()
        engine.stop()

        safe_serial_write(b'3')  # speaking end

    except Exception as e:
        print("TTS Error:", e)

# ---------------- SPEECH RECOGNITION ---------------- #
recognizer = sr.Recognizer()

def listen():
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("Listening...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)

        text = recognizer.recognize_google(audio)
        print("User:", text)
        return text

    except sr.WaitTimeoutError:
        print("Listening timeout")
    except:
        print("Could not understand.")

    return None

# ---------------- LLM ---------------- #
def ask_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Reply in ONE short sentence only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,
        max_tokens=60
    )
    return response.choices[0].message.content.strip()

# ---------------- EYE BLINK ---------------- #
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

EYE_THRESH = 0.25
EYE_CONSEC_FRAMES = 5
eye_counter = 0

detector_dlib = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# ---------------- HEAD ROTATION ---------------- #
HEAD_ROTATION_THRESHOLD = 40
head_rotation_sent = False

def get_head_pose(shape, frame_size):
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])

    image_points = np.array([
        shape[30], shape[8], shape[36],
        shape[45], shape[48], shape[54]
    ], dtype="double")

    focal_length = frame_size[1]
    center = (frame_size[1] / 2, frame_size[0] / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    _, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    return angles

# ---------------- VOICE THREAD ---------------- #
voice_thread = None
voice_lock = threading.Lock()

def voice_interaction(emotion):
    global voice_thread

    try:
        prompt = f"The user looks {emotion}. Reply in ONE short sentence only."
        reply = ask_llm(prompt)
        print("AI:", reply)
        speak(reply)

        user_input = listen()
        if user_input:
            follow_up = ask_llm(user_input)
            print("AI:", follow_up)
            speak(follow_up)

    except Exception as e:
        print("Voice thread error:", e)

    finally:
        with voice_lock:
            voice_thread = None

# ---------------- MAIN ---------------- #
def main(args):
    global eye_counter, voice_thread, head_rotation_sent

    model = Mini_Xception().to(device)
    model.eval()
    checkpoint = torch.load(args.pretrained, map_location=device)
    model.load_state_dict(checkpoint['mini_xception'])

    face_alignment = FaceAlignment()
    face_detector = HaarCascadeDetector('face_detector') if args.haar else DnnDetector('face_detector')

    video = cv2.VideoCapture(0)

    stable_emotion = None
    emotion_counter = 0
    triggered = False
    STABLE_THRESHOLD = 5
    cooldown_time = 10
    last_trigger_time = 0

    safe_serial_write(b'4')

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detector_dlib(gray, 0)

        # ---- Eye + Head ---- #
        for subject in subjects:
            shape = predictor(gray, subject)
            shape = face_utils.shape_to_np(shape)

            pitch, yaw, roll = get_head_pose(shape, frame.shape)

            if abs(yaw) > HEAD_ROTATION_THRESHOLD:
                safe_serial_write(b'1')
            else:
                head_rotation_sent = False

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

            if ear < EYE_THRESH:
                eye_counter += 1
                if eye_counter >= EYE_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                    safe_serial_write(b'5')
            else:
                eye_counter = 0
                safe_serial_write(b'4')

        # ---- Emotion ---- #
        faces = face_detector.detect_faces(frame)
        for face in faces:
            (x, y, w, h) = face

            input_face = face_alignment.frontalize_face(face, frame)
            input_face = cv2.resize(input_face, (48,48))
            input_face = histogram_equalization(input_face)

            input_face = transforms.ToTensor()(input_face).to(device)
            input_face = torch.unsqueeze(input_face, 0)

            with torch.no_grad():
                pred = model(input_face)
                emotion = get_label_emotion(torch.argmax(pred).item())

            if emotion == stable_emotion:
                emotion_counter += 1
            else:
                stable_emotion = emotion
                emotion_counter = 1
                triggered = False

            if (emotion_counter >= STABLE_THRESHOLD and not triggered and
                time.time() - last_trigger_time > cooldown_time):

                if voice_thread is None:
                    voice_thread = threading.Thread(
                        target=voice_interaction,
                        args=(stable_emotion,)
                    )
                    voice_thread.start()

                triggered = True
                last_trigger_time = time.time()

            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
            cv2.putText(frame, emotion, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

        cv2.imshow("AI Assistant", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    video.release()
    cv2.destroyAllWindows()

    if ser:
        ser.close()

# ---------------- ENTRY ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--haar', action='store_true')
    parser.add_argument('--pretrained', type=str,
                        default='checkpoint/model_weights/weights_epoch_75.pth.tar')
    args = parser.parse_args()

    main(args)
