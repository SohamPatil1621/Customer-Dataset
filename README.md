# Customer-Dataset
A Customer Dataset Project in Python that analyzes customer data to uncover insights like purchase patterns, customer segmentation, and retention trends. Using pandas, matplotlib, and scikit-learn, it helps businesses make data-driven decisions to improve marketing and customer service strategies.

import os
import shutil
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
# Define dataset paths
part1 = "/content/drive/MyDrive/part1"
part2 = "/content/drive/MyDrive/part2"
metadata_csv = "/content/drive/MyDrive/ham10000_dataset/HAM10000_metadata.csv"
merged_folder = "/content/drive/MyDrive/ham10000_dataset/HAM10000_merged_images"

# Create Merged Folder if it doesn't exist
os.makedirs(merged_folder, exist_ok=True)

# Copy Images from Part 1 & Part 2
for folder in [part1, part2]:
    if os.path.exists(folder):  # Check if folder exists before copying
        for file in os.listdir(folder):
            src_path = os.path.join(folder, file)
            dst_path = os.path.join(merged_folder, file)
            if not os.path.exists(dst_path):  # Avoid duplicate copies
                shutil.copy(src_path, dst_path)

print(f" Merged folder created at: {merged_folder}")
print(f"Total images in merged folder: {len(os.listdir(merged_folder))}")
# Load metadata for labels
df = pd.read_csv(metadata_csv)
df = df[['image_id', 'dx']]  # Keep only image ID and diagnosis
df['image_id'] = df['image_id'] + ".jpg"  # Add .jpg extension
df = df.set_index('image_id')  # Set image_id as index
# Define image size & batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
# Create directories for each class
output_dir = "/content/ham10000_dataset"
os.makedirs(output_dir, exist_ok=True)
# Create subfolders for each label
classes = df['dx'].unique()
for c in classes:
    os.makedirs(os.path.join(output_dir, c), exist_ok=True)
    # Move images into respective class folders
for image in os.listdir(merged_folder):
    if image in df.index:
        label = df.loc[image, 'dx']
        shutil.move(os.path.join(merged_folder, image), os.path.join(output_dir, label, image))
import os
import shutil
import pandas as pd

# Google Drive Paths
dataset_dir = "/content/drive/MyDrive"
part1 = os.path.join(dataset_dir, "part1")
part2 = os.path.join(dataset_dir, "part2")
metadata_path = os.path.join(dataset_dir, "HAM10000_metadata.csv")
merged_folder = os.path.join(dataset_dir, "HAM10000_organized")

# Create Merged Folder if it doesn't exist
os.makedirs(merged_folder, exist_ok=True)

# Load Metadata File
df = pd.read_csv(metadata_path)

# Iterate through rows to organize images
for _, row in df.iterrows():
    image_name = row['image_id'] + ".jpg"  # Image filename
    class_label = row['dx']  # Class label (e.g., 'mel', 'nv', etc.)

    # Create class folder inside merged folder
    class_folder = os.path.join(merged_folder, class_label)
    os.makedirs(class_folder, exist_ok=True)

    # Find the image in part1 or part2 and move it
    src_path = None
    for folder in [part1, part2]:
        potential_path = os.path.join(folder, image_name)
        if os.path.exists(potential_path):
            src_path = potential_path
            break

    # Move the image if found
    if src_path:
        dst_path = os.path.join(class_folder, image_name)
        shutil.move(src_path, dst_path)

print(f"Images have been organized into class folders at: {merged_folder}")
import os

organized_folder = "/content/drive/MyDrive/HAM10000_organized"

# Check if the organized dataset exists
if os.path.exists(organized_folder):
    print(f" Organized dataset found at: {organized_folder}")
    print("Classes available:")
    for class_name in os.listdir(organized_folder):
        class_path = os.path.join(organized_folder, class_name)
        print(f"{class_name}")
else:
    print(" Organized dataset NOT found!")
    #  Define dataset path (already organized by class)
dataset_path = "/content/drive/MyDrive/HAM10000_organized"
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 80% train, 20% validation
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
train_generator = datagen.flow_from_directory(
    organized_folder,
    target_size=(256, 256),  #  Must match model input
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Use subset to split data
)

val_generator = datagen.flow_from_directory(
    organized_folder,
    target_size=(256, 256),  #  Must match model input
    batch_size=32,
    class_mode='categorical',
    subset='validation'  #  Use subset to split data
)
# Load validation data
val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)
from tensorflow.keras.layers import GlobalAveragePooling2D

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    GlobalAveragePooling2D(),  #  Replaces Flatten to dynamically adjust shape
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

