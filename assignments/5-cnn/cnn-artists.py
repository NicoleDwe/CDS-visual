#!/usr/bin/env python

"""
This script builds a deep learning model using LeNet as the convolutional neural network architecture. This network is used to classify impressionist paintings by their artists. 

Usage:
    $ python cnn-artists.py
"""

### DEPENDENCIES ###

# Data tools
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import glob
import argparse
from contextlib import redirect_stdout

# Sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# TensorFlow tools
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K



### MAIN FUNCTION ###

def main():
    
    print("\n[INFO] Initializing the construction of a LeNet convolutional neural network model...")
    
    
    ### ARGPARSE ###
    
    # Initialize parser
    ap = argparse.ArgumentParser()
    
    # Path to training data
    ap.add_argument("-tr", "--train_data",
                    type = str,
                    required = False,
                    help = "Path to the training data",
                    default = "../../data/impressionist_subset/training/training")
    
    # Path to test data
    ap.add_argument("-te", "--test_data",
                    type = str,
                    required = False,
                    help = "Path to the test/validation data",
                    default = "../../data/impressionist_subset/validation/validation")
    
    # Number of epochs
    ap.add_argument("-e", "--n_epochs",
                    type = int,
                    required = False,
                    help = "The number of epochs to train the model on",
                    default = 20)
    
    # Batch size
    ap.add_argument("-b", "--batch_size",
                    type = int,
                    required = False,
                    help = "The size of the batch on which to train the model",
                    default = 32)
    
    # parse arguments
    args = vars(ap.parse_args())
    
    # save input parameters
    train_data = args["train_data"]
    test_data = args["test_data"]
    n_epochs = args["n_epochs"]
    batch_size = args["batch_size"]
    
    
    ### PREPARE OUTPUT DIRECTORY ###
    
    # Create output directory
    if not os.path.exists(os.path.join("out")):
        os.mkdir(os.path.join("out"))
        
        
    ### PREPARE DATA ###
    
    print("\n[INFO] Preparing data...")
    
    # Get label names from directory names
    label_names = get_labels(train_data)
    
    # Find minimum dimension to resize image accordingly
    min_dimension = get_min_dimension(train_data, test_data, label_names)
    print(f"\n[INFO] Input images will be resized to dimensions of {min_dimension}x{min_dimension}...")

    # Create X and Y for train and test data
    trainX, trainY = create_XY(train_data, min_dimension, label_names)
    testX, testY = create_XY(test_data, min_dimension, label_names)
    
    # Normalize data and binarize labels
    trainX, trainY, testX, testY = normalize_binarize(trainX, trainY, testX, testY)
    
    
    ### TRAINING LENET MODEL ###
    
    print(f"\n[INFO] Initializing LeNet Model with {n_epochs} and batch size of {batch_size}...")
    
    # Defining model
    model = define_LeNet_model(min_dimension)
    # Training model
    H = train_LeNet_model(model, trainX, trainY, testX, testY, n_epochs, batch_size)
    
    
    ### EVALUATING MODEL ###
    
    print(f"\n[INFO] Evaluating LeNet model...")
    
    # Plot loss/accuracy history of the model
    plot_history(H, n_epochs)
    
    # Evaluate model
    print("\n[INFO] Evaluating model... Below is the classification report. This can also be found in the out folder.\n")
    evaluate_model(model, testX, testY, batch_size, label_names)
    
    # User message
    print("\n[INFO] Done! Information about the model, classification report and plot of the model history can be found in 'out/' directory.\n")
     


    
### FUNCTIONS  ###

def get_labels(path):
    """
    Define the label names by listing the names of the folders within training directory without listing hidden files. 
    """
    # Create empty list
    label_names = []
    
    # For every name in training directory
    for name in os.listdir(path):
        # If it does not start with . (which hidden files do)
        if not name.startswith('.'):
            label_names.append(name)
            
    return label_names


def get_min_dimension(train_data, test_data, label_names):
    """
    Function to estimate the minimum dimension across all images (and their heights and widths). 
    The minimum dimension is later used to resize the image to a square of this minimum dimension. 
    """
    # Create empty lists
    dimensions = []
    
    # For each artist
    for name in label_names: 
        # Get the test and train images
        images = glob.glob(os.path.join(train_data, name, "*.jpg")) + glob.glob(os.path.join(test_data, name, "*.jpg"))
        # For each image
        for image in images:
            # Load the image
            loaded_img = cv2.imread(image)
            # Append the dimensions to the heights and widths list 
            dimensions.append(loaded_img.shape[0])
            dimensions.append(loaded_img.shape[1])
            
    # Find the min values in the list
    min_dimension = min(dimensions)
    
    return min_dimension


def create_XY(data, min_dimension, label_names):
    """
    This function creates and array of images (X), and a list with the corresponding labels (Y)
    """
    # Empty array for images (X) and list for labels (Y)
    X = np.empty((0, min_dimension, min_dimension, 3))
    Y = []
    
    # Loop through the labels (artist names)
    for name in label_names:
        images = glob.glob(os.path.join(data, name, "*.jpg"))
        
        # For each image for the given artist
        for image in tqdm(images):
            
            # Load image
            loaded_img = cv2.imread(image)
            # Resize image with the specified dimensions
            resized_img = cv2.resize(loaded_img, (min_dimension, min_dimension), interpolation = cv2.INTER_AREA)
            # Create array of image
            image_array = np.array([np.array(resized_img)])
            # Append to array annd list
            X = np.vstack((X, image_array))
            Y.append(name)
        
    return X, Y


def normalize_binarize(trainX, trainY, testX, testY):
    """
    This function normalizes the training and test data and binarizes the training and test labels. 
    """
    
    # Normalize training and test data
    trainX_norm = trainX.astype("float") / 255.
    testX_norm = testX.astype("float") / 255.
    
    # Binarize training and test labels
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)
    
    return trainX_norm, trainY, testX_norm, testY


def define_LeNet_model(min_dimension):
    """
    This function defines the LeNet model architecture.
    """
    # Define model
    model = Sequential()

    # First set of layers (convolutional, activation, max pooling)
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(min_dimension, min_dimension, 3))) # Convolutional layer
    model.add(Activation("relu")) # Activation Function 
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) # Max pooling layer: stride of 2 horizontal, 2 vertical
    
    # Second set of layers (convolutional, activation, max pooling)
    model.add(Conv2D(50, (5, 5), padding="same")) # Convolutional layer
    model.add(Activation("relu")) # Activation function
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) # Max pooling layer
    
    # Add fully-connected layer
    model.add(Flatten()) # flattening layer
    model.add(Dense(500)) # dense network with 500 nodes
    model.add(Activation("relu")) # activation function
    
    # Add output layer, softmax classifier
    model.add(Dense(10)) # dense layer of 10 nodes used to classify the images
    model.add(Activation("softmax"))

    # Define optimizer 
    opt = SGD(lr=0.01)
    
    # Compile model
    model.compile(loss="categorical_crossentropy", 
                  optimizer=opt, 
                  metrics=["accuracy"])
    
    # Save model summary
    with open('out/model_summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
    
    # Visualization of model
    plot_model(model, to_file = "out/LeNet_model.png", show_shapes=True, show_layer_names=True)
    
    return model


def train_LeNet_model(model, trainX, trainY, testX, testY, n_epochs, batch_size):
    """
    This function trains the LeNet model on the training data and validates it on the test data.
    """
    # Train model
    H = model.fit(trainX, trainY, 
                  validation_data=(testX, testY), 
                  batch_size=batch_size, 
                  epochs=n_epochs, verbose=1)
    
    return H
    
    
def plot_history(H, n_epochs):
    """
    Function that plots the loss/accuracy of the model during training.
    """
    # Visualize performance
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, n_epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, n_epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, n_epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, n_epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("out/model_history.png")
    
    
def evaluate_model(model, testX, testY, batch_size, label_names):
    """
    This function evaluates the trained model and saves the classification report in output directory. 
    """
    # Predictions
    predictions = model.predict(testX, batch_size=batch_size)
    
    # Classification report
    classification = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names)
            
    # Print classification report
    print(classification)
    
    # Save classification report
    with open("out/classification_report.txt", 'w', encoding='utf-8') as f:
        f.writelines(classification_report(testY.argmax(axis=1),
                                                  predictions.argmax(axis=1),
                                                  target_names=label_names))

        
# Define behaviour when called from command line
if __name__=="__main__":
    main()