import cv2, pickle, os, numpy as np, pandas as pd
from datetime import datetime, timedelta
import speech_recognition as sr
import pyttsx3
from deepface import DeepFace
from mtcnn import MTCNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "models", "face_embeddings.pkl")
LOG_FILE = os.path.join(BASE_DIR, "logs", "attendance_log.csv")

with open(EMBEDDINGS_FILE, "rb") as f:
    face_db = pickle.load(f)

engine = pyttsx3.init()
speak_engine = lambda t: (engine.say(t), engine.runAndWait())

detector = MTCNN()
face_detector = MTCNN()
recognizer = sr.Recognizer()
mic = sr.Microphone()

marked = {}
last_seen = {}
silent_timer = datetime.now()

if not os.path.exists(LOG_FILE):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    pd.DataFrame(columns=["Name","Entry Time","Exit Time","Confidence"]).to_csv(LOG_FILE, index=False)

def match_face(embedding):
    best, min_dist = "Unknown", float("inf")
    for name, embeddings in face_db.items():
        for db_emb in embeddings:
            dist = np.linalg.norm(np.array(db_emb) - np.array(embedding))
            if dist < min_dist:
                min_dist, best = dist, name
    conf = max(15, 100 - min_dist * 10)
    return best, round(conf, 2)

def listen_voice():
    global silent_timer
    try:
        with mic as src:
            recognizer.adjust_for_ambient_noise(src, 0.2)
            audio = recognizer.listen(src, timeout=2, phrase_time_limit=3)
        text = recognizer.recognize_google(audio).lower()
        silent_timer = datetime.now()
        if "mark attendance for" in text:
            name = text.replace("mark attendance for","").strip().title()
            if name in face_db and name not in marked:
                ts = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                df = pd.read_csv(LOG_FILE)
                df.loc[len(df)] = [name, ts, "", 0]
                df.to_csv(LOG_FILE,index=False)
                marked[name]=True
                speak_engine(f"Attendance marked for {name}")
    except:
        pass

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        speak_engine("Camera not accessible")
        break

    frame = cv2.flip(frame, 1)
    faces = face_detector.detect_faces(frame)
    now = datetime.now()

    if now - silent_timer > timedelta(seconds=7):
        for face_obj in faces:
            box = face_obj.get("box", None)
            if not box:
                continue
            x,y,w,h = box
            face_crop = frame[y:y+h, x:x+w]
            try:
                emb = DeepFace.represent(face_crop, "Facenet", enforce_detection=False)[0]["embedding"]
                name, conf = match_face(emb)
                if name != "Unknown" and name not in marked:
                    entry = now.strftime("%d-%m-%Y %H:%M:%S")
                    df = pd.read_csv(LOG_FILE)
                    df.loc[len(df)] = [name, entry, "", conf]
                    df.to_csv(LOG_FILE,index=False)
                    marked[name]=True
                    last_seen[name]=now
                    speak_engine(f"{name} attendance auto marked")
            except:
                pass
        silent_timer = now

    for face_obj in faces:
        box = face_obj.get("box", None)
        if not box:
            continue
        x,y,w,h = box
        face_crop = frame[y:y+h, x:x+w]
        try:
            emb = DeepFace.represent(face_crop, "Facenet", enforce_detection=False)[0]["embedding"]
            name, conf = match_face(emb)
            if name != "Unknown":
                last_seen[name] = now
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
            cv2.putText(frame,f"{name} ({conf}%)",(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        except:
            pass

    overlay_y = 30
    cv2.putText(frame,"Present (IN):",(10,overlay_y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    for i, name in enumerate(marked.keys()):
        cv2.putText(frame,f"{name}",(10,overlay_y+25*(i+1)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

    for name in list(last_seen.keys()):
        if last_seen[name] and now - last_seen[name] > timedelta(seconds=6):
            exit_time = now.strftime("%d-%m-%Y %H:%M:%S")
            df = pd.read_csv(LOG_FILE)
            df.loc[(df["Name"]==name)&(df["Exit Time"].isna()), "Exit Time"] = exit_time
            df.to_csv(LOG_FILE,index=False)
            speak_engine(f"{name} has exited")
            last_seen[name]=None

    listen_voice()
    cv2.imshow("Attendance Dashboard", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
