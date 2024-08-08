import cv2
import numpy as np
import os

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the LBPH model for face recognition
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Function to load training images and labels
# For example, you can have a "training_data" folder with subfolders for each person.
# Each subfolder contains several images of the person.
def load_training_data(data_folder_path):
    images = []
    labels = []
    label_names = []
    label_id = 0

    for person_name in os.listdir(data_folder_path):
        person_folder_path = os.path.join(data_folder_path, person_name)
        label_names.append(person_name)

        for image_name in os.listdir(person_folder_path):
            image_path = os.path.join(person_folder_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            images.append(image)
            labels.append(label_id)

        label_id += 1

    return images, labels, label_names

# Load training data
training_data_folder = r"C:\Users\amalg\Desktop\Projet\training_data"
images, labels, label_names = load_training_data(training_data_folder)

# Train the recognizer model
recognizer.train(images, np.array(labels))

# Function to detect and recognize faces in an image
def detect_and_recognize_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        label_id, confidence = recognizer.predict(face)
        label_name = label_names[label_id]

        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Display the person's name and confidence
        cv2.putText(image, f"{label_name} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return image

# Open a webcam and apply face detection and recognition in real-time
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_and_recognize_faces(frame)
    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
