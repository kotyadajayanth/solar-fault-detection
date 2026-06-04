import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────
DATASET_PATH = r"C:\Users\Admin\OneDrive\Desktop\Projects\solar-fault-detection\dataset"
MODEL_SAVE_PATH = r"C:\Users\Admin\OneDrive\Desktop\Projects\solar-fault-detection\model\solar_model.keras"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30

# ─── DATA LOADING ─────────────────────────────────────────
print("Loading dataset...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.3,
    shear_range=0.2,
    brightness_range=[0.7, 1.3]
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"\nClasses found: {train_generator.class_indices}")
NUM_CLASSES = len(train_generator.class_indices)

# ─── TRANSFER LEARNING MODEL (MobileNetV2) ────────────────
print("\nBuilding Transfer Learning model...")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model first
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─── CALLBACKS ────────────────────────────────────────────
checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

# ─── PHASE 1: Train top layers ─────────────────────────────
print("\nPhase 1: Training top layers...")

history1 = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop]
)

# ─── PHASE 2: Fine-tune last 30 layers ────────────────────
print("\nPhase 2: Fine-tuning...")

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop]
)

# ─── COMBINE HISTORY ──────────────────────────────────────
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

# ─── RESULTS ──────────────────────────────────────────────
print(f"\nTraining Complete!")
print(f"Best Validation Accuracy: {max(val_acc)*100:.2f}%")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(acc, label='Train Accuracy')
ax1.plot(val_acc, label='Val Accuracy')
ax1.set_title('Model Accuracy')
ax1.legend()

ax2.plot(loss, label='Train Loss')
ax2.plot(val_loss, label='Val Loss')
ax2.set_title('Model Loss')
ax2.legend()

plt.tight_layout()
plt.savefig(r"C:\Users\Admin\OneDrive\Desktop\Projects\solar-fault-detection\model\training_results.png")
plt.show()
print("Training graph saved!")