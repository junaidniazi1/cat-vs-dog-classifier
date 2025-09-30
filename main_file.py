

"""# CAT VS DOG

#Step 1 — Download & inspect dataset (Colab + Kaggle)



Install Kaggle and upload your kaggle.json (API token).
Run this cell, upload your kaggle.json when prompted:
"""

import os, shutil
from google.colab import files

# Make directory
os.makedirs('/root/.kaggle', exist_ok=True)

# Upload kaggle.json
uploaded = files.upload()

# Move kaggle.json to the correct path
shutil.move("kaggle.json", "/root/.kaggle/kaggle.json")

# Fix file permissions
os.chmod("/root/.kaggle/kaggle.json", 0o600)
print("Kaggle API token installed successfully!")

"""## Download and unzip the dataset from Kaggle:"""

!kaggle datasets download -d bhavikjikadara/dog-and-cat-classification-dataset -p /content --unzip

"""## List files and inspect folder structure:"""

# 4. Inspect dataset files and folders
import os, glob, pandas as pd

root = '/content/PetImages'  # dataset path in Colab
print("Listing files/folders in /content (first 200 entries):")
for i, p in enumerate(sorted(os.listdir(root))):
    if i < 200:
        print("-", p)

# Try to find common image folders inside extracted dataset
possible_dirs = []
for dirpath, dirnames, filenames in os.walk(root):
    # limit depth and size of printing
    if any(fname.lower().endswith(('.jpg','.jpeg','.png')) for fname in filenames):
        possible_dirs.append(dirpath)

print("\nDirectories containing images (sample):")
for d in possible_dirs[:10]:
    print("-", d)

"""## Count images per folder and show a few sample images (visual check):"""

# 5. Count images & display a sample from each class
import matplotlib.pyplot as plt
from PIL import Image
import random

# find folders that look like classes (have images)
image_dirs = [d for d in possible_dirs if len(os.listdir(d))>0]
# we'll pick top-level dirs that likely contain 'cats' or 'dogs'
print("Image-containing directories found (count):", len(image_dirs))

# Create dictionary of counts
counts = {}
for d in image_dirs:
    imgs = [f for f in os.listdir(d) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    counts[d] = len(imgs)

# show counts (top 20)
for d, c in sorted(counts.items(), key=lambda x: -x[1])[:20]:
    print(f"{d}: {c} images")

# Display random sample images (up to 4)
samples = []
for d, c in list(counts.items())[:4]:
    imgs = [os.path.join(d,f) for f in os.listdir(d) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if imgs:
        samples.append(imgs[random.randrange(len(imgs))])

print("\nDisplaying up to 4 sample images:")
plt.figure(figsize=(12,4))
for i,p in enumerate(samples):
    try:
        img = Image.open(p).convert('RGB')
        plt.subplot(1, len(samples), i+1)
        plt.imshow(img)
        plt.title(os.path.basename(os.path.dirname(p)))
        plt.axis('off')
    except Exception as e:
        print("Failed to open", p, e)
plt.show()

"""## (Optional quick check) Print image sizes distribution to see if resizing will be needed:"""

# 6. Image size distribution (sample up to 200 images to speed up)
from collections import Counter
sizes = Counter()
checked = 0
for d in image_dirs:
    for fname in os.listdir(d):
        if fname.lower().endswith(('.jpg','.jpeg','.png')):
            try:
                im = Image.open(os.path.join(d,fname))
                sizes[im.size] += 1
            except:
                pass
            checked += 1
            if checked >= 200:
                break
    if checked >= 200:
        break

print("Sample image size counts (width, height):")
for s,c in sizes.most_common():
    print(s, c)

"""#Step 2 — Preprocessing + Train/Validation/Test Split + Data Augmentation

We’ll use TensorFlow/Keras ImageDataGenerator to:

Split into train/validation/test

Apply data augmentation (rotation, flip, zoom, brightness, etc.)

Normalize pixel values (0–1)

✅ What this does:

Resizes all images to 128×128.

Applies augmentation only to training data.

Splits dataset into 80% training / 20% validation.

Normalizes pixel values (0–1).
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ------------------------------
# Paths (adjust if dataset unzipped differently)
# ------------------------------
# Example: if dataset has two folders: 'cats' and 'dogs' inside a folder
dataset_dir = "/content/PetImages"  # <- change to your dataset folder name

# ------------------------------
# Image parameters
# ------------------------------
img_size = (128, 128)   # resize images
batch_size = 32

# ------------------------------
# Data Augmentation for training
# ------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% for validation
)

# Only rescaling for validation & test
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# ------------------------------
# Train & Validation Generators
# ------------------------------
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
    shuffle=True
)

# ------------------------------
# OPTIONAL: Separate Test Split
# ------------------------------
# If dataset has no explicit "test" folder, we can split validation into two parts later.
# For now, we keep train/validation.

"""# Step 3 — Build CNN Model (with Conv Layers, Pooling, Dropout, BatchNorm)

We’ll design a moderately deep CNN for Cats vs Dogs classification.


✅ What this model has:

3 convolutional blocks → deeper spatial features.

MaxPooling layers → reduce dimensions.

Dropout → prevent overfitting.

BatchNormalization → stabilize training.

Dense layer + sigmoid → binary classification (cats vs dogs).
"""

from tensorflow.keras import layers, models

# ------------------------------
# CNN Model
# ------------------------------
model = models.Sequential([
    # 1st Convolution Block
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Dropout(0.25),

    # 2nd Convolution Block
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Dropout(0.25),

    # 3rd Convolution Block
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Dropout(0.3),

    # Flatten + Dense Layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    # Output Layer (binary classification: cat vs dog)
    layers.Dense(1, activation='sigmoid')
])

# ------------------------------
# Compile Model
# ------------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Summary
model.summary()

"""# Step 4 — Train the CNN (with Early Stopping & Model Checkpointing)

This step will:

Train the CNN on train/validation data.

Use EarlyStopping (stop when validation accuracy stops improving).

Save the best model weights with ModelCheckpoint.

📊 What this does:

Trains for up to 25 epochs (but may stop early if no improvement).

Automatically reduces learning rate when stuck.

Saves the best model as best_cnn_model.h5.

Restores best weights at the end.
"""

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ------------------------------
# Callbacks
# ------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,        # stop if no improvement for 5 epochs
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_cnn_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

# ------------------------------
# Training
# ------------------------------
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

"""# 🔹 Step 5 — Evaluate Model + Visualize Results + Confusion Matrix

This step will:

Plot training vs validation accuracy/loss curves.

Evaluate final accuracy on the validation set.

Show a confusion matrix with predictions.

Display a few sample predictions (cat/dog).


✅ What you’ll get:

Graphs: Accuracy & loss curves across epochs.

Validation Accuracy (final model performance).

Confusion Matrix: See where cats/dogs are misclassified.

Classification Report: Precision, Recall, F1-score.

Sample Predictions: 6 random images with predicted vs actual labels.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ------------------------------
# 1. Plot Training Curves
# ------------------------------
plt.figure(figsize=(12,5))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend(); plt.grid(); plt.title("Accuracy")

# Loss plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.grid(); plt.title("Loss")

plt.show()

# ------------------------------
# 2. Evaluate Final Model
# ------------------------------
val_loss, val_acc = model.evaluate(val_generator)
print(f"\n✅ Validation Accuracy: {val_acc:.4f}")
print(f"✅ Validation Loss: {val_loss:.4f}")

# ------------------------------
# 3. Confusion Matrix
# ------------------------------
# Get true labels & predictions
val_generator.reset()
y_true = val_generator.classes
y_pred = (model.predict(val_generator) > 0.5).astype("int32")

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Cat","Dog"], yticklabels=["Cat","Dog"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Cat","Dog"]))

# ------------------------------
# 4. Show Sample Predictions
# ------------------------------
import random
from tensorflow.keras.preprocessing import image

sample_images, sample_labels = next(val_generator)

plt.figure(figsize=(12,6))
for i in range(6):
    idx = random.randint(0, len(sample_images)-1)
    img = sample_images[idx]
    true_label = "Dog" if sample_labels[idx] == 1 else "Cat"
    pred_prob = model.predict(img[np.newaxis,...])[0][0]
    pred_label = "Dog" if pred_prob > 0.5 else "Cat"

    plt.subplot(2,3,i+1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"True: {true_label}\nPred: {pred_label} ({pred_prob:.2f})")

plt.show()

"""#🔹 Step 6 — Save, Load & Test on Custom Images

This step will let you:

Save your trained CNN model.

Load it later without retraining.

Upload your own cat/dog images in Colab and get predictions.
"""

from tensorflow.keras.models import load_model

# Save the trained model
model.save("cats_vs_dogs_cnn.h5")
print("✅ Model saved as cats_vs_dogs_cnn.h5")

# Load model later (for inference or deployment)
loaded_model = load_model("cats_vs_dogs_cnn.h5")
print("✅ Model loaded successfully")

"""# 📌 Code (Test on Your Own Images)"""

from google.colab import files
import numpy as np
from tensorflow.keras.preprocessing import image

# Upload custom image(s)
uploaded = files.upload()

for fn in uploaded.keys():
    # Load image
    img_path = fn
    img = image.load_img(img_path, target_size=(128, 128))  # resize same as training
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Prediction
    pred_prob = loaded_model.predict(img_array)[0][0]
    pred_label = "Dog 🐶" if pred_prob > 0.5 else "Cat 🐱"

    print(f"\nImage: {fn}")
    print(f"Prediction: {pred_label} (Confidence: {pred_prob:.2f})")

    # Show image
    plt.imshow(image.load_img(img_path))
    plt.axis("off")
    plt.title(pred_label)
    plt.show()

