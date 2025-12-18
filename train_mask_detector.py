import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam as LegacyAdam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import numpy as np
import os
import cv2

# --- Configuration ---
INIT_LR = 1e-4          # Initial learning rate
EPOCHS = 20
BS = 32                # Batch size
DATASET_PATH = "dataset/fmd_training_data"
MODEL_PATH = "models/mask_detector.h5"

# --- Data Preparation ---
def prepare_data(dataset_path):
    print("Loading images and labels...")
    data = []
    labels = []
    
    # Iterate through each person's folder
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        # 1. Unmasked Faces (Label: 'unmasked')
        for filename in os.listdir(person_folder):
            img_path = os.path.join(person_folder, filename)
            if not os.path.isdir(img_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    image = cv2.imread(img_path)
                    image = cv2.resize(image, (224, 224)) # Resize for MobileNetV2
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    data.append(image)
                    labels.append("unmasked")
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

        # 2. Masked Faces (Label: 'masked')
        masked_folder = os.path.join(person_folder, "masked")
        if os.path.exists(masked_folder):
             for filename in os.listdir(masked_folder):
                img_path = os.path.join(masked_folder, filename)
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        image = cv2.imread(img_path)
                        image = cv2.resize(image, (224, 224))
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        data.append(image)
                        labels.append("masked")
                    except Exception as e:
                        pass

    # Convert to NumPy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    # Preprocessing
    data = data / 255.0

    # Binarize the labels (masked: 0, unmasked: 1 or vice-versa)
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    # Split data
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
        test_size=0.20, stratify=labels, random_state=42)
    
    return trainX, testX, trainY, testY, lb

# --- Model Building (MobileNetV2 Transfer Learning) ---
def build_model(width, height, depth, classes):
    # Load the MobileNetV2 network, ensuring the head FC layer sets are left off
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(height, width, depth)))

    # Construct the new head of the model that will be placed on top of the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(classes, activation="softmax")(headModel)

    # Place the head FC model on top of the base model
    model = Model(inputs=baseModel.input, outputs=headModel)

    # Freeze the layers in the base model so they won't be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False

    return model

# --- Main Execution ---
if __name__ == "__main__":
    trainX, testX, trainY, testY, lb = prepare_data(DATASET_PATH)
    
    # Initialize the training data augmentation object
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

    # Build and compile the model
    model = build_model(224, 224, 3, len(lb.classes_))
    opt = Adam(learning_rate=INIT_LR)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Train the head of the network
    print("[INFO] training head...")
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS)

    # Evaluate and save the model
    print("[INFO] evaluating network...")
    predIdxs = model.predict(testX, batch_size=BS)
    predIdxs = np.argmax(predIdxs, axis=1)

    print(classification_report(testY.argmax(axis=1), predIdxs,
        target_names=lb.classes_))

    # Ensure the models folder exists and save the model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"[INFO] saving mask detector model to {MODEL_PATH}...")
    model.save(MODEL_PATH, save_format="h5")