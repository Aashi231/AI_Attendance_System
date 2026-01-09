# from deepface import DeepFace
# import os
# import pickle

# face_db = {}

# # Loop through each person folder in dataset
# for person in os.listdir("dataset"):
#     embeddings = []
#     person_path = os.path.join("dataset", person)

#     for img in os.listdir(person_path):
#         img_path = os.path.join(person_path, img)

#         # Generate embedding for each image
#         try:
#             rep = DeepFace.represent(img_path, model_name="ArcFace", enforce_detection=False)
#             embeddings.append(rep[0]["embedding"])
#             print(f"Encoded {person}/{img}")
#         except Exception as e:
#             print(f"Error encoding {img_path}: {e}")

#     face_db[person] = embeddings

# # Save database
# with open("face_db.pkl", "wb") as f:
#     pickle.dump(face_db, f)

# print("Face embedding database created successfully!")
# print("Encoding completed. Embeddings saved in models/")


#new code below

import os
import pickle
from deepface import DeepFace

# -------------------------------
# Set paths relative to this script
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EMBEDDINGS_FILE = os.path.join(MODELS_DIR, "face_embeddings.pkl")
FAILED_LOG_FILE = os.path.join(MODELS_DIR, "failed_images.txt")

# Create models folder if it doesn't exist
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# -------------------------------
# Dictionary to hold embeddings
# -------------------------------
face_db = {}

# Clear previous failed log if exists
if os.path.exists(FAILED_LOG_FILE):
    os.remove(FAILED_LOG_FILE)

# -------------------------------
# Loop through each person in dataset
# -------------------------------
for person_name in os.listdir(DATASET_DIR):
    person_folder = os.path.join(DATASET_DIR, person_name)
    
    if not os.path.isdir(person_folder):
        continue  # skip files if any

    embeddings_list = []

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        try:
            embedding = DeepFace.represent(
                img_path=image_path, 
                model_name="Facenet",
                enforce_detection=True  # keep True for accuracy
            )[0]["embedding"]
            embeddings_list.append(embedding)
            print(f"Encoded {person_name}/{image_name}")
        except Exception as e:
            print(f"Error encoding {person_name}/{image_name}: {e}")
            # Log failed image to file
            with open(FAILED_LOG_FILE, "a") as log_file:
                log_file.write(f"{person_name}/{image_name}\n")

    if embeddings_list:
        face_db[person_name] = embeddings_list

# -------------------------------
# Save embeddings to file
# -------------------------------
with open(EMBEDDINGS_FILE, "wb") as f:
    pickle.dump(face_db, f)

print("Face embedding database created successfully!")
print(f"Embeddings saved in {EMBEDDINGS_FILE}")
print(f"Failed images logged in {FAILED_LOG_FILE}")

