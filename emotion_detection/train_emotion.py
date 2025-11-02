import os
import cv2
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_dataset(main_folder_path, target_size=(48, 48), grayscale=True):
    emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise']
    train_X, train_y, test_X, test_y = [], [], [], []

    def process_image(image, target_size, grayscale):
        if grayscale and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, target_size)
        return image

    for folder_type, X_list, y_list in [('train', train_X, train_y), ('test', test_X, test_y)]:
        path = os.path.join(main_folder_path, folder_type)
        if not os.path.exists(path):
            continue
        for emotion in emotion_classes:
            emotion_path = os.path.join(path, emotion)
            if os.path.exists(emotion_path):
                image_files = [f for f in os.listdir(emotion_path)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for img_file in image_files:
                    img_path = os.path.join(emotion_path, img_file)
                    image = cv2.imread(img_path)
                    if image is not None:
                        X_list.append(process_image(image, target_size, grayscale))
                        y_list.append(emotion)

    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)


main_folder_path = r"datasets\FER2013"

print("🚀 Loading dataset...")
train_X, train_y, test_X, test_y = load_dataset(main_folder_path, target_size=(48, 48), grayscale=True)

# Reshape for CNN input
train_X = train_X.reshape(-1, 48, 48, 1).astype('float32') / 255.0
test_X = test_X.reshape(-1, 48, 48, 1).astype('float32') / 255.0

# Encode labels
encoder = LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.transform(test_y)

train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

num_classes = train_y.shape[1]

print(f"✅ Dataset loaded: {train_X.shape[0]} train, {test_X.shape[0]} test samples.")


datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True
)
datagen.fit(train_X)


def build_cnn(input_shape=(48,48,1), num_classes=7):
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),

        Conv2D(256, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_cnn()
model.summary()


callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_emotion_model.h5', monitor='val_accuracy', save_best_only=True)
]


history = model.fit(
    datagen.flow(train_X, train_y, batch_size=64),
    validation_data=(test_X, test_y),
    epochs=1000,
    callbacks=callbacks,
    verbose=1
)


test_loss, test_acc = model.evaluate(test_X, test_y, verbose=0)
print(f"\n🎯 Test Accuracy: {test_acc*100:.2f}%")

# Save final model
model.save('final_emotion_cnn.h5')
print("💾 Model saved as 'final_emotion_cnn.h5'")




import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.show()
