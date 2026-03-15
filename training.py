"""
Emotion Classification CNN Model Training
This script trains a convolutional neural network to classify emotions (7 classes)
using grayscale facial images.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import (Dense, Input, Dropout, GlobalAveragePooling2D, 
                          Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D)
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
picture_size = 48
batch_size = 128
epochs = 48
num_classes = 7
learning_rate = 0.001
initial_learning_rate = 0.0001

# Path to dataset
folder_path = "images/"  # Update this path to your dataset location

# Emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

print("="*70)
print("EMOTION CLASSIFICATION CNN MODEL - TRAINING SCRIPT")
print("="*70)
print(f"Image Size: {picture_size}x{picture_size}")
print(f"Batch Size: {batch_size}")
print(f"Epochs: {epochs}")
print(f"Number of Classes: {num_classes}")
print(f"Learning Rate: {learning_rate}")
print("="*70)

# ============================================================================
# 2. VERIFY DATASET PATHS
# ============================================================================
if not os.path.exists(folder_path):
    print(f"❌ ERROR: Dataset folder '{folder_path}' not found!")
    print(f"Current working directory: {os.getcwd()}")
    exit()

train_path = os.path.join(folder_path, "train")
validation_path = os.path.join(folder_path, "validation")

if not os.path.exists(train_path) or not os.path.exists(validation_path):
    print(f"❌ ERROR: Train or validation folder not found!")
    print(f"Expected: {train_path}")
    print(f"Expected: {validation_path}")
    exit()

print("✅ Dataset paths verified successfully!")

# ============================================================================
# 3. DATA LOADING AND AUGMENTATION
# ============================================================================
print("\n📊 Loading and preparing data...")

# Training data generator (with augmentation)
datagen_train = ImageDataGenerator()

# Validation data generator (no augmentation, just normalization)
datagen_val = ImageDataGenerator()

# Load training data
train_set = datagen_train.flow_from_directory(
    train_path,
    target_size=(picture_size, picture_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Load validation data
test_set = datagen_val.flow_from_directory(
    validation_path,
    target_size=(picture_size, picture_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

print(f"✅ Training samples: {train_set.n}")
print(f"✅ Validation samples: {test_set.n}")
print(f"✅ Classes: {train_set.class_indices}")

# ============================================================================
# 4. MODEL ARCHITECTURE
# ============================================================================
print("\n🏗️  Building CNN model architecture...")

model = Sequential()

# 1st Convolutional Block
model.add(Conv2D(64, (3, 3), padding='same', input_shape=(picture_size, picture_size, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2nd Convolutional Block
model.add(Conv2D(128, (5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 3rd Convolutional Block
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 4th Convolutional Block
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flatten layer
model.add(Flatten())

# 1st Fully Connected Layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

# 2nd Fully Connected Layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Output Layer
model.add(Dense(num_classes, activation='softmax'))

print("✅ Model architecture created successfully!")

# ============================================================================
# 5. MODEL COMPILATION
# ============================================================================
print("\n⚙️  Compiling model...")

optimizer = Adam(learning_rate=initial_learning_rate)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
print("\n" + "="*70)
print("MODEL ARCHITECTURE SUMMARY")
print("="*70)
model.summary()
print("="*70)

# ============================================================================
# 6. CALLBACKS SETUP
# ============================================================================
print("\n📋 Setting up training callbacks...")

# Checkpoint callback - saves the best model
checkpoint = ModelCheckpoint(
    "./model.h5",
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

# Early stopping - stops training if validation loss doesn't improve
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=3,
    verbose=1,
    restore_best_weights=True
)

# Learning rate reduction - reduces learning rate if validation loss plateaus
reduce_learning_rate = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    verbose=1,
    min_delta=0.0001
)

callbacks_list = [early_stopping, checkpoint, reduce_learning_rate]

print("✅ Callbacks configured:")
print("   - ModelCheckpoint: Saves best model based on validation accuracy")
print("   - EarlyStopping: Stops if validation loss doesn't improve for 3 epochs")
print("   - ReduceLROnPlateau: Reduces learning rate if validation loss plateaus")

# ============================================================================
# 7. MODEL TRAINING
# ============================================================================
print("\n" + "="*70)
print("🚀 STARTING MODEL TRAINING")
print("="*70)

history = model.fit(
    train_set,
    steps_per_epoch=train_set.n // train_set.batch_size,
    epochs=epochs,
    validation_data=test_set,
    validation_steps=test_set.n // test_set.batch_size,
    callbacks=callbacks_list,
    verbose=1
)

print("\n" + "="*70)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("="*70)

# ============================================================================
# 8. PLOT TRAINING RESULTS
# ============================================================================
print("\n📈 Plotting training results...")

plt.style.use('dark_background')
plt.figure(figsize=(20, 10))

# Plot 1: Loss
plt.subplot(1, 2, 1)
plt.suptitle('Model Performance - Optimizer: Adam', fontsize=16, y=1.02)
plt.ylabel('Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.legend(loc='upper right', fontsize=12)
plt.grid(True, alpha=0.3)

# Plot 2: Accuracy
plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("✅ Plot saved as 'training_history.png'")

plt.show()

# ============================================================================
# 9. FINAL STATISTICS
# ============================================================================
print("\n" + "="*70)
print("TRAINING STATISTICS")
print("="*70)
final_train_loss = history.history['loss'][-1]
final_train_acc = history.history['accuracy'][-1]
final_val_loss = history.history['val_loss'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print(f"Final Training Loss: {final_train_loss:.4f}")
print(f"Final Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
print(f"Final Validation Loss: {final_val_loss:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
print("="*70)

print("\n✅ Model saved as 'model.h5'")
print("🎉 Training complete! You can now use the model for inference with main.py")
