# ============================================================
# NSDC VIRTUAL INTERNSHIP
# Virtual Data Science with Python
# WEEK 5 PROJECT
# Deep Learning Application in Data Science
# Trainee: Mozammil Islam
#
# Project: MNIST Handwritten Digit Classification
# Framework: TensorFlow / Keras
# Model: Convolutional Neural Network (CNN)
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


# ============================================================
# 2. RANDOM SEED FOR REPRODUCIBILITY
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# 3. LOAD MNIST DATASET
# ============================================================

print("\n================ LOADING MNIST DATASET ================\n")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training Images Shape:", x_train.shape)
print("Training Labels Shape:", y_train.shape)

print("Testing Images Shape:", x_test.shape)
print("Testing Labels Shape:", y_test.shape)


# ============================================================
# 4. DATASET INFORMATION
# ============================================================

print("\n================ DATASET INFORMATION ================\n")

print("Number of Training Images:", len(x_train))
print("Number of Testing Images:", len(x_test))

print("Image Height:", x_train.shape[1])
print("Image Width:", x_train.shape[2])

print("Number of Classes:", len(np.unique(y_train)))

print(
    "Classes:",
    np.unique(y_train)
)


# ============================================================
# 5. DISPLAY SAMPLE IMAGES
# ============================================================

plt.figure(figsize=(10, 5))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(
        x_train[i],
        cmap="gray"
    )

    plt.title(
        f"Label: {y_train[i]}"
    )

    plt.axis("off")

plt.suptitle(
    "Sample MNIST Handwritten Digits"
)

plt.tight_layout()

plt.show()


# ============================================================
# 6. NORMALIZE PIXEL VALUES
# ============================================================

# Original pixel values range from 0 to 255.
# Convert them to the range 0 to 1.

x_train = (
    x_train.astype("float32")
    / 255.0
)

x_test = (
    x_test.astype("float32")
    / 255.0
)

print(
    "\nPixel values normalized from 0-255 to 0-1."
)


# ============================================================
# 7. ADD CHANNEL DIMENSION
# ============================================================

# CNN requires input shape:
# (samples, height, width, channels)

x_train = np.expand_dims(
    x_train,
    axis=-1
)

x_test = np.expand_dims(
    x_test,
    axis=-1
)

print("\nShape After Adding Channel Dimension:")

print(
    "x_train:",
    x_train.shape
)

print(
    "x_test:",
    x_test.shape
)


# ============================================================
# 8. CHECK CLASS DISTRIBUTION
# ============================================================

unique_classes, class_counts = np.unique(
    y_train,
    return_counts=True
)

class_distribution = pd.DataFrame({

    "Digit": unique_classes,

    "Training Samples": class_counts
})

print(
    "\n================ CLASS DISTRIBUTION ================\n"
)

print(class_distribution)


# ============================================================
# 9. BUILD CNN MODEL
# ============================================================

model = Sequential([

    # First Convolution Layer
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),

    # First Pooling Layer
    MaxPooling2D(
        (2, 2)
    ),

    # Second Convolution Layer
    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    # Second Pooling Layer
    MaxPooling2D(
        (2, 2)
    ),

    # Convert feature maps into a vector
    Flatten(),

    # Fully connected layer
    Dense(
        128,
        activation="relu"
    ),

    # Dropout for regularization
    Dropout(
        0.30
    ),

    # Output layer
    Dense(
        10,
        activation="softmax"
    )
])


# ============================================================
# 10. DISPLAY MODEL ARCHITECTURE
# ============================================================

print(
    "\n================ CNN MODEL SUMMARY ================\n"
)

model.summary()


# ============================================================
# 11. COMPILE MODEL
# ============================================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)

print(
    "\nModel compiled successfully."
)


# ============================================================
# 12. EARLY STOPPING
# ============================================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True
)


# ============================================================
# 13. TRAIN CNN MODEL
# ============================================================

print(
    "\n================ TRAINING CNN MODEL ================\n"
)

history = model.fit(

    x_train,
    y_train,

    epochs=15,

    batch_size=128,

    validation_split=0.10,

    callbacks=[
        early_stop
    ],

    verbose=1
)


# ============================================================
# 14. TRAINING HISTORY
# ============================================================

print(
    "\nTraining completed."
)

print(
    "Epochs Actually Trained:",
    len(history.history["loss"])
)


# ============================================================
# 15. TRAINING VS VALIDATION ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "Training vs Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 16. TRAINING VS VALIDATION LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Training vs Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 17. EVALUATE MODEL ON TEST DATA
# ============================================================

print(
    "\n================ TEST SET EVALUATION ================\n"
)

test_loss, test_accuracy = model.evaluate(

    x_test,
    y_test,

    verbose=0
)

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.4f}"
)


# ============================================================
# 18. GENERATE PREDICTIONS
# ============================================================

y_probability = model.predict(

    x_test,

    verbose=0
)

y_pred = np.argmax(

    y_probability,

    axis=1
)

print(
    "\nPredictions generated successfully."
)


# ============================================================
# 19. CLASSIFICATION REPORT
# ============================================================

print(
    "\n================ CLASSIFICATION REPORT ================\n"
)

report = classification_report(

    y_test,

    y_pred,

    digits=4
)

print(report)


# ============================================================
# 20. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,

    y_pred
)

print(
    "\n================ CONFUSION MATRIX ================\n"
)

print(cm)


# ============================================================
# 21. CONFUSION MATRIX VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 8))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.title(
    "MNIST Confusion Matrix"
)

plt.tight_layout()

plt.show()


# ============================================================
# 22. DISPLAY SAMPLE PREDICTIONS
# ============================================================

plt.figure(figsize=(12, 8))

for i in range(12):

    plt.subplot(
        3,
        4,
        i + 1
    )

    plt.imshow(
        x_test[i].squeeze(),
        cmap="gray"
    )

    plt.title(
        f"True: {y_test[i]} | Pred: {y_pred[i]}"
    )

    plt.axis("off")

plt.suptitle(
    "Sample MNIST Predictions"
)

plt.tight_layout()

plt.show()


# ============================================================
# 23. ERROR ANALYSIS
# ============================================================

incorrect_indices = np.where(

    y_pred != y_test

)[0]


print(
    "\n================ ERROR ANALYSIS ================\n"
)

print(
    "Total Misclassified Images:",
    len(incorrect_indices)
)

print(
    "Total Correctly Classified Images:",
    len(y_test) - len(incorrect_indices)
)


# ============================================================
# 24. DISPLAY MISCLASSIFIED IMAGES
# ============================================================

if len(incorrect_indices) > 0:

    plt.figure(
        figsize=(12, 8)
    )

    display_count = min(
        12,
        len(incorrect_indices)
    )

    for i in range(display_count):

        index = incorrect_indices[i]

        plt.subplot(
            3,
            4,
            i + 1
        )

        plt.imshow(
            x_test[index].squeeze(),
            cmap="gray"
        )

        plt.title(
            f"True: {y_test[index]}\n"
            f"Pred: {y_pred[index]}"
        )

        plt.axis("off")

    plt.suptitle(
        "Misclassified MNIST Images"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 25. CALCULATE ERROR RATE
# ============================================================

error_rate = (

    len(incorrect_indices)

    / len(y_test)

)

print(
    f"\nError Rate: {error_rate:.4f}"
)


# ============================================================
# 26. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(

    history.history

)

history_df.to_csv(

    "training_history.csv",

    index=False

)

print(
    "\nTraining history saved as:"
)

print(
    "training_history.csv"
)


# ============================================================
# 27. SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "Actual_Digit": y_test,

    "Predicted_Digit": y_pred,

    "Correct": (
        y_test == y_pred
    )

})

prediction_df.to_csv(

    "mnist_predictions.csv",

    index=False

)

print(
    "Predictions saved as:"
)

print(
    "mnist_predictions.csv"
)


# ============================================================
# 28. SAVE CLASSIFICATION REPORT
# ============================================================

report_dict = classification_report(

    y_test,

    y_pred,

    output_dict=True
)

report_df = pd.DataFrame(

    report_dict

).transpose()

report_df.to_csv(

    "classification_report.csv"
)

print(
    "Classification report saved as:"
)

print(
    "classification_report.csv"
)


# ============================================================
# 29. FINAL MODEL SUMMARY
# ============================================================

print(
    "\n======================================================"
)

print(
    "WEEK 5 DEEP LEARNING PROJECT COMPLETED SUCCESSFULLY"
)

print(
    "======================================================"
)

print(
    "\nModel: Convolutional Neural Network (CNN)"
)

print(
    "Dataset: MNIST Handwritten Digits"
)

print(
    "Input Shape: 28 x 28 x 1"
)

print(
    "Output Classes: 10"
)

print(
    f"Test Accuracy: {test_accuracy:.4f}"
)

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Misclassified Images: {len(incorrect_indices)}"
)

print(
    f"Error Rate: {error_rate:.4f}"
)
