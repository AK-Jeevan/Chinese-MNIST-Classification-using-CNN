# Chinese MNIST: 15,000 grayscale images of handwritten Chinese numerals (0–14) for multiclass image classification using CNN.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Rescaling
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# Step 1: Read data
data = pd.read_csv(r"C:\Users\akjee\Documents\AI\DL\CNN\Chinese_MNIST\chinese_mnist.csv")
print("Raw label values:", data['value'].unique())

# Step 2: Map all unique values to 0...N-1 (label encoding)
data['label_enc'], uniques = pd.factorize(data['value'])
print("Unique encoded labels:", np.unique(data['label_enc']))
print("Mapping (encoded -> original):")
for idx, original in enumerate(uniques):
    print(f"{idx} -> {original}")

# Step 3: File paths
image_dir = r"C:\Users\akjee\Documents\AI\DL\CNN\Chinese_MNIST\Images"
data['filename'] = data.apply(
    lambda row: f"input_{row['suite_id']}_{row['sample_id']}_{row['code']}.jpg", axis=1
)
data['filepath'] = data['filename'].apply(lambda x: f"{image_dir}\\{x}")

# Step 4: Split data
X_paths = data['filepath'].values
y_labels = data['label_enc'].values  # NEW: use encoded labels

train_paths, test_paths, train_labels, test_labels = train_test_split(
    X_paths, y_labels, test_size=0.2, random_state=42, stratify=y_labels
)

def load_images(paths, labels, img_size=(64, 64)):
    images = []
    for path in paths:
        img = tf.keras.utils.load_img(path, color_mode='grayscale', target_size=img_size)
        img = tf.keras.utils.img_to_array(img) / 255.0
        images.append(img)
    images = np.array(images)
    labels = np.array(labels, dtype=np.int32)
    return images, labels

X_train, y_train = load_images(train_paths, train_labels)
X_test, y_test = load_images(test_paths, test_labels)

print("Unique labels in training set:", np.unique(y_train))
print("Unique labels in test set:", np.unique(y_test))

num_classes = len(uniques)

# CNN model
model = Sequential([
    Rescaling(1., input_shape=(64, 64, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPool2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # dynamically adapts!
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.2
)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.3f}")

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training History')
plt.show()
