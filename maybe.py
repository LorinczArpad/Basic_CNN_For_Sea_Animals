# main.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import seaborn as sns
# Configuration
BATCH_SIZE = 32
TARGET_SIZE = (224, 224)
IMG_HEIGHT, IMG_WIDTH = TARGET_SIZE
DATASET_PATH = "./input/sea-animals-image-dataste"
TEST_IMAGE_PATH = "./test_images/coral.jpg"
MODEL_PATH = "sea_animals_classifier.keras"

def prepare_data(dataset_path):
    # Get file paths and labels
    image_dir = Path(dataset_path)
    filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + \
                list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.PNG'))
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')
    image_df = pd.concat([filepaths, labels], axis=1)

    # Split into train and test datasets
    train_df, test_df = train_test_split(image_df, test_size=0.2, shuffle=True, random_state=42)
    return train_df, test_df

def create_generators(train_df, test_df):
    train_gen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        validation_split=0.2,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)

    train_images = train_gen.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        seed=42
    )
    val_images = train_gen.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        seed=42
    )
    test_images = test_gen.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    return train_images, val_images, test_images

def build_model(num_classes):
    base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def predict_image(model, class_labels, image_path):
    img = tf.keras.utils.load_img(image_path, target_size=TARGET_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_label = class_labels[np.argmax(score)]
    confidence = 100 * np.max(score)
    return predicted_label, confidence
def create_plot(history):
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(accuracy))
    # Plot and save the accuracy graph
    plt.figure()
    plt.plot(epochs, accuracy, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig('accuracy_plot.png')  # Save as accuracy_plot.png
    plt.close()

    # Plot and save the loss graph
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_plot.png')  # Save as loss_plot.png
    plt.close()
def main():
    if tf.config.list_physical_devices('GPU'):
        print("GPU is available!")
    else:
        print("GPU is not available!")
    train_df, test_df = prepare_data(DATASET_PATH)
    train_images, val_images, test_images = create_generators(train_df, test_df)

    class_labels = list(test_images.class_indices.keys())
    print("Class labels:", class_labels)

    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded.")
    else:
        model = build_model(num_classes=len(class_labels))
        history = model.fit(
                train_images,
                validation_data=val_images,
                epochs=30,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
                    tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)
                ]
            )
        model.save('anyad.keras')
        create_plot(history)
        print("Model trained and saved.")

    model.summary()

    predicted_label, confidence = predict_image(model, class_labels, TEST_IMAGE_PATH)
    print(f"This image most likely belongs to {predicted_label} with {confidence:.2f}% confidence.")

if __name__ == "__main__":
    main()
