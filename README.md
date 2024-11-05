# Import necessary libraries

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.optimizers import SGD

from tensorflow.keras.datasets import mnist

# Load the MNIST dataset

(trainX, trainY), (testX, testY) = mnist.load_data()

# Flatten the images to be a simple list of 28x28=784 pixels

trainX = trainX.reshape((trainX.shape[0], 28 * 28)).astype("float32") / 255.0

testX = testX.reshape((testX.shape[0], 28 * 28)).astype("float32") / 255.0

# Convert the labels from integers to vectors for one-hot encoding

lb = LabelBinarizer()

trainY = lb.fit_transform(trainY)

testY = lb.transform(testY)

# Define the model architecture

model = Sequential()

model.add(Dense(256, input_shape=(784,), activation="relu"))

model.add(Dense(128, activation="relu"))

model.add(Dense(10, activation="softmax"))

# Compile the model

sgd = SGD(learning_rate=0.01)

model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])


# Train the model
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=50, batch_size=128)
# Evaluate the network
predictions = model.predict(testX)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1)))
# Plot training and validation loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 50), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 50), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
# Plot only accuracy over epochs
plt.figure()
plt.plot(np.arange(0, 50), H.history["accuracy"], label="train_accuracy", color="blue")
plt.plot(np.arange(0, 50), H.history["val_accuracy"], label="val_accuracy", color="orange")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Plot the loss ratio (train_loss / val_loss) over epochs

loss_ratio = np.array(H.history["loss"]) / np.array(H.history["val_loss"])

plt.figure()

plt.plot(np.arange(0, 50), loss_ratio, label="Loss Ratio (train/val)", color="purple")

plt.title("Training vs Validation Loss Ratio")

plt.xlabel("Epoch #")

plt.ylabel("Loss Ratio")

plt.legend()

plt.show()

# Assuming a constant learning rate from SGD optimizer

learning_rate = 0.01 # The current learning rate

# Plot the learning rate as a constant line

plt.figure()

plt.plot(np.arange(0, 50), [learning_rate] * 50, label="Learning Rate", color="green")

plt.title("Learning Rate Over Epochs")

plt.xlabel("Epoch #")

plt.ylabel("Learning Rate")

plt.legend()

plt.show()
