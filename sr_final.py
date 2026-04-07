"""
Emotion + Eye Blink + Voice AI Assistant + Serial Signals
Author: Amr Elsersy
Updated: Serial communication + Head Rotation integrated
Fix: pyttsx3 re-initialized per call to work correctly across threads
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

# ---------------- TTS (Thread-Safe) ---------------- #
# pyttsx3 CANNOT be shared across threads.
# Solution: create a fresh engine inside every speak() call.
def speak(text):
    if ser:
        try:
            ser.write(b'2')   # speaking start signal
        except:
            pass

    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)
        engine.setProperty('volume', 1.0)
        engine.say(text)
        engine.runAndWait()
        engine.stop()          # <-- cleanly release the engine
    except Exception as e:
        # Fallback to PowerShell TTS if pyttsx3 fails
        print(f"pyttsx3 failed ({e}), falling back to PowerShell TTS")
        safe_text = text.replace("'", "")
        command = (
            f'powershell -Command "Add-Type -AssemblyName System.Speech; '
            f'$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
            f"$speak.Speak('{safe_text}');\"" 
        )
        os.system(command)

    if ser:
        try:
            ser.write(b'3')   # speaking end signal
        except:
            pass

        
# ---------------- SPEECH RECOGNITION ---------------- #
recognizer = sr.Recognizer()

def listen():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print("User:", text)
        return text
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
        shape[30],
        shape[8],
        shape[36],
        shape[45],
        shape[48],
        shape[54]
    ], dtype="double")

    focal_length = frame_size[1]
    center = (frame_size[1] / 2, frame_size[0] / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    pitch, yaw, roll = angles
    return pitch, yaw, roll

# ---------------- VOICE THREAD ---------------- #
voice_thread = None
voice_lock = threading.Lock()

def voice_interaction(emotion):
    global voice_thread
    prompt = f"The user looks {emotion}. Reply in ONE short sentence only."
    reply = ask_llm(prompt)
    print("AI:", reply)
    speak(reply)            # <-- now creates its own engine, safe in any thread

    user_input = listen()
    if user_input:
        follow_up = ask_llm(user_input)
        print("AI:", follow_up)
        speak(follow_up)    # <-- same fix applies here

    with voice_lock:
        voice_thread = None

# ---------------- MAIN ---------------- #
def main(args):
    global eye_counter, voice_thread, head_rotation_sent

    mini_xception = Mini_Xception().to(device)
    mini_xception.eval()
    checkpoint = torch.load(args.pretrained, map_location=device)
    mini_xception.load_state_dict(checkpoint['mini_xception'])

    face_alignment = FaceAlignment()
    face_detector = HaarCascadeDetector('face_detector') if args.haar else DnnDetector('face_detector')

    video = cv2.VideoCapture(0)

    stable_emotion = None
    emotion_counter = 0
    triggered = False
    STABLE_THRESHOLD = 5
    cooldown_time = 10
    last_trigger_time = 0

    t1 = 0
    ser.write(b'4')
    sts = 1

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detector_dlib(gray, 0)

        # -------- Eye + Head Detection -------- #
        for subject in subjects:
            shape = predictor(gray, subject)
            shape = face_utils.shape_to_np(shape)

            # ---- HEAD ROTATION ----
            pitch, yaw, roll = get_head_pose(shape, frame.shape)

            cv2.putText(frame, f"Yaw: {round(yaw,1)}", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            if abs(yaw) > HEAD_ROTATION_THRESHOLD:
                cv2.putText(frame, "HEAD ROTATION ALERT!", (10,90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

                if ser:
                    try:
                        if not head_rotation_sent:
                            ser.write(b'1')
                            head_rotation_sent = True
                    except:
                        pass
            else:
                head_rotation_sent = False

            # ---- EYE BLINK ----
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0),1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0),1)

            if ear < EYE_THRESH:
                eye_counter += 1
                if eye_counter >= EYE_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                    if ser:
                        try:
                            if(sts==1):
                                sts=0
                                ser.write(b'5')
                        except:
                            pass
            else:
                eye_counter = 0
                if ser:
                    try:
                        if(sts==0):
                            sts=1
                            ser.write(b'4')
                    except:
                        pass

        # -------- Emotion Detection -------- #
        faces = face_detector.detect_faces(frame)
        for face in faces:
            (x, y, w, h) = face
            input_face = face_alignment.frontalize_face(face, frame)
            input_face = cv2.resize(input_face, (48,48))
            input_face = histogram_equalization(input_face)

            input_face = transforms.ToTensor()(input_face).to(device)
            input_face = torch.unsqueeze(input_face, 0)

            with torch.no_grad():
                emotion_pred = mini_xception(input_face)
                emotion_index = torch.argmax(emotion_pred).item()
                emotion = get_label_emotion(emotion_index)

            if emotion == stable_emotion:
                emotion_counter += 1
            else:
                stable_emotion = emotion
                emotion_counter = 1
                triggered = False

            if (emotion_counter >= STABLE_THRESHOLD and not triggered and 
                time.time() - last_trigger_time > cooldown_time):
                if voice_thread is None:
                    voice_thread = threading.Thread(target=voice_interaction, args=(stable_emotion,))
                    voice_thread.start()
                triggered = True
                last_trigger_time = time.time()

            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
            cv2.putText(frame, emotion, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

        t2 = time.time()
        fps = round(1/(t2-t1+1e-5))
        t1 = t2
        cv2.putText(frame, f"FPS: {fps}", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        cv2.imshow("Emotion + Eye Blink + Head Rotation AI Assistant", frame)
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
