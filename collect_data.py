import cv2
import os

# Ask for person's name
name = input("Enter person name: ").strip()

# Create folder automatically
dataset_path = os.path.join("dataset", name)
os.makedirs(dataset_path, exist_ok=True)

cap = cv2.VideoCapture(0)

print("Camera started.")
print("Press 'c' to capture image")
print("Press 'q' to quit")

count = 0
MAX_IMAGES = 20

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not accessible")
        break

    cv2.imshow("Face Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and count < MAX_IMAGES:
        img_path = os.path.join(dataset_path, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        count += 1
        print(f"Captured image {count}/{MAX_IMAGES}")

    elif key == ord('q') or count == MAX_IMAGES:
        break

cap.release()
cv2.destroyAllWindows()

print("Dataset collection complete.")
