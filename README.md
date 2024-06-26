# mnist_digit_detection
MNIST Digit Detection Using DL

# MNIST Digit Detection using ANN

## Overview
This project aims to detect handwritten digits from the MNIST dataset using an Artificial Neural Network (ANN). The dataset consists of 70,000 images, each of dimension 28x28, representing digits from 0 to 9.

## Dataset
The MNIST dataset contains 70,000 images with the following structure:
- 60,000 images for training
- 10,000 images for testing

Each image is a 28x28 grayscale image, and each label is a digit from 0 to 9.

## Data Preprocessing
1. **Load Dataset**:
   - Loaded the dataset using `tensorflow.keras.datasets.mnist.load_data()`.

2. **Normalization**:
   - Normalized the pixel values of the images by dividing by 255:
     ```python
     X_train, X_test = X_train / 255.0, X_test / 255.0
     ```

## Model Training
1. **Model Definition**:
   - Defined the Sequential model using `tf.keras.models.Sequential` and added layers using `Dense` and `Dropout` from `tf.keras.layers`:
     ```python
     from tensorflow.keras.models import Sequential
     from tensorflow.keras.layers import Dense, Dropout, Flatten, Input

     model = Sequential()
     model.add(Input(shape=(28, 28)))
     model.add(Flatten())
     model.add(Dense(784, activation="relu"))
     model.add(Dropout(0.3))
     model.add(Dense(392, activation="relu"))
     model.add(Dropout(0.3))
     model.add(Dense(196, activation="relu"))
     model.add(Dropout(0.3))
     model.add(Dense(98, activation="relu"))
     model.add(Dropout(0.3))
     model.add(Dense(49, activation="relu"))
     model.add(Dropout(0.3))
     model.add(Dense(24, activation="relu"))
     model.add(Dropout(0.3))
     model.add(Dense(10, activation="sigmoid"))

     model.compile(optimizer="adam",
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
     ```

2. **Model Training**:
   - Trained the model for 10 epochs with the following command:
     ```python
     model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=10)
     ```

## Model Evaluation
1. **Accuracy and Loss**:
   - Evaluated the model on the test set:
     ```plaintext
     Accuracy: 0.9737
     Loss: 0.1513
     ```

2. **Confusion Matrix**:
   - Computed the confusion matrix for detailed performance analysis:
     ```plaintext
     [[ 968, 1, 1, 0, 1, 0, 4, 1, 2, 2],
      [ 0, 1123, 3, 3, 0, 2, 2, 1, 1, 0],
      [ 4, 0, 999, 16, 2, 0, 2, 6, 3, 0],
      [ 0, 0, 0, 997, 0, 4, 0, 4, 2, 3],
      [ 2, 1, 2, 0, 964, 0, 5, 0, 0, 8],
      [ 3, 0, 0, 12, 1, 869, 3, 0, 2, 2],
      [ 2, 3, 0, 1, 10, 424, 516, 0, 2, 0],
      [ 0, 2, 12, 1, 0, 0, 0, 1003, 2, 8],
      [ 6, 1, 3, 20, 8, 12, 0, 4, 903, 17],
      [ 1, 4, 0, 4, 10, 3, 0, 2, 1, 984]]
     ```

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/riitk/mnist_digit_detection.git
   cd mnist_digit_detection

## Acknowledgements
- The dataset used in this project was obtained from the MNIST dataset.
- Thank you to the open-source community for providing valuable resources and tools.
