import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pathlib import Path
import matplotlib.pyplot as plt


# Define emotion classes (same order as your training)
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise']

# Load your best model
model_path = Path(r"models\final_emotion_cnn.h5")
model = load_model(model_path)

def preprocess_image(img_path, target_size=(48, 48)):
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Error: unable to load image {img_path}")

    # Convert to grayscale (since model trained on grayscale)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, target_size)
    img = img.reshape(1, 48, 48, 1).astype('float32') / 255.0
    return img

def predict_emotion(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    predicted_class = emotion_classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    print(f"Predicted Emotion: {predicted_class} ({confidence:.2f}%)")

    # Display the image
    img_display = cv2.imread(img_path)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    plt.imshow(img_display)
    plt.axis('off')
    plt.title(f"{predicted_class} ({confidence:.1f}%)")
    plt.show()

import os
test_path = r"test-predict"
for i in range(5):
    image_test = "test" + str(i) + ".jpg"
    image_path = os.path.join(test_path, image_test)
    predict_emotion(image_path)
