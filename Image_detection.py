import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load pre-trained model
model = load_model('image_detection_model.h5')

# Define labels
labels = ['Cat', 'Dog', 'Car', 'Person', 'Bicycle', 'Traffic Light', 'Truck', 'Bus']

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0  # Normalize
    return np.expand_dims(image, axis=0)

def predict_image(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return labels[class_index], confidence

# Test with an example image
image_path = 'test_image.jpg'
predicted_label, confidence = predict_image(image_path)
print(f'Detected Object: {predicted_label} (Confidence: {confidence:.2f}%)')

def live_video_detection():
    cap = cv2.VideoCapture(0)  # Open webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (128, 128))
        normalized_frame = resized_frame / 255.0
        input_frame = np.expand_dims(normalized_frame, axis=0)
        prediction = model.predict(input_frame)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        label = labels[class_index]
        cv2.putText(frame, f'{label} ({confidence:.2f}%)', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Live Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Function to process a video file
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (128, 128))
        normalized_frame = resized_frame / 255.0
        input_frame = np.expand_dims(normalized_frame, axis=0)
        prediction = model.predict(input_frame)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        label = labels[class_index]
        cv2.putText(frame, f'{label} ({confidence:.2f}%)', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Video Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
