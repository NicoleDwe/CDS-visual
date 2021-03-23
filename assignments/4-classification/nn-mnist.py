#!/usr/bin/env python

"""
Training a neural network classifier, printing and saving evalutation metrics 

For the mnist data:
    - download the images and labels of the mnist data
    - preprocess and scale the data to be in an appropriate format
    - train a neural network classifier based on training data
    - evaluate the performance of the classifier using the test data 
    - print performance metrics on command line and save to a txt file
Input parameters:
    - (optional) output_filename: str <output_filename>
Example: 
    python3 lr-mnist.py --output_filename out_metrics.txt
Output: 
    - performance metrics saved in output file (default: nn_metrics.txt)
"""

# import dependencies 
import os 
import sys
import numpy as np
sys.path.append(os.path.join("..", ".."))

# utils for network
from utils.neuralnetwork import NeuralNetwork

# sci-kit learn
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

# argparse
import argparse

def main(): 
    
    ### ARGUMENT PARSER ###
    # argparse for input parameters
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output_filename", help = "Name of the output file of metrics", default="nn_metrics.txt")
    args = vars(ap.parse_args())
    output_filename = args["output_filename"]
    
    ### INPUT DATA ###
    print("[INFO] Getting MNIST data ...")
    images, labels = fetch_openml('mnist_784', version=1, return_X_y=True)
    
    ### NN CLASSIFIER ###   
    print("[INFO] Initialising neural network ...")
    nn_classifier = NN_classifier(images, labels)
    images_train, images_test, labels_train, labels_test = nn_classifier.prepare_data(test_size=0.2)
    nn_model = nn_classifier.train_network(images_train, labels_train)
    nn_metrics = nn_classifier.evaluate_network(nn_model, images_test, labels_test)
    nn_classifier.save_metrics(nn_model, nn_metrics, output_filename)
    
    ## OUTPUT ###
    print(f"[OUTPUT] Perfomance metrics of the nn-classifier:\n{nn_metrics}")        
    print(f"[INFO] All done, file with metrics saved in out/{output_filename}") 
    
    
class NN_classifier:
    
    def __init__(self, images, labels):
        
        # images and labels
        self.images, self.labels = images, labels
    
    def prepare_data(self, test_size):
        """
        Preparing data: turning into array, floats and scaling with min-max regularisation
        Splitting data: into test, training data
        Binarising labels for neural network training
        """
        # turn into array
        images = np.array(self.images.astype("float"))
        labels = np.array(self.labels)
        # min-max regularisation
        scaled_images = (images - images.min())/(images.max() - images.min())
        # splitting data
        images_train, images_test, labels_train, labels_test = train_test_split(
            scaled_images, labels, random_state=9, test_size=test_size)
        # binarising labels
        labels_train_binary = LabelBinarizer().fit_transform(labels_train)
        labels_test_binary = LabelBinarizer().fit_transform(labels_test)
        return images_train, images_test, labels_train_binary, labels_test_binary
    
    def train_network(self, images_train, labels_train_binary):
        """
        Training network with hidden layers
        """
        nn = NeuralNetwork([images_train.shape[1], 32, 16, 10])
        nn.fit(images_train, labels_train_binary, epochs=10) 
        return nn
    
    def evaluate_network(self, nn, images_test, labels_test_binary):
        """
        Evaluating network based on predictions on test-data
        """
        # predicting labels
        predictions = nn.predict(images_test)
        predictions = predictions.argmax(axis=1)
        # getting performance metrics
        nn_metrics = classification_report(labels_test_binary.argmax(axis=1), predictions)
        return nn_metrics
    
    def save_metrics(self, nn_classifier, nn_metrics, output_filename):
        """
        Saving performance metric in txt file in defined output path
        """
        # if the directory does not exist, create it 
        if not os.path.exists("out"):
            os.mkdir("out")
        # define path
        output_filepath = os.path.join("out", output_filename)
        # save metrics to path
        with open(output_filepath, "w") as output_file:
            output_file.write(f"Output for {nn_classifier}:\n\nClassification Metrics:\n{nn_metrics}")

            
if __name__=="__main__":
    main()