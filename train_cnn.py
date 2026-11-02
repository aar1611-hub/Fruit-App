
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Basic settings
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
DATASET_PATH = "fruit_dataset"

# 2. Load training images
train_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_names=["apple", "banana", "orange"]
)

# 3. Load validation images
validation_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_names=["apple", "banana", "orange"]
)

print("Class names:", train_data.class_names)

# 4. Build the CNN
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),

    # Convert pixel values from 0-255 to 0-1
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    # Three output classes
    layers.Dense(3, activation="softmax")
])

# 5. Configure the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 6. Train the model
model.fit(
    train_data,
    validation_data=validation_data,
    epochs=10
)

# 7. Save the trained model
model.save("fruit_cnn.keras")

print("CNN training completed!")
print("Model saved as fruit_cnn.keras")
