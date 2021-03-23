#!/usr/bin/env python

"""
Training a logistic regression classifier, printing and saving evalutation metrics 

For the mnist data:
    - download the images and labels of the mnist data
    - preprocess and scale the data to be in an appropriate format
    - train a logistic regression classifier based on training data
    - evaluate the performance of the classifier using the test data 
    - print performance metrics on command line and save to a txt file
Input parameters:
    - (optional) output_filename: str <output_filename>
Example: 
    python3 lr-mnist.py --output_filename out_metrics.txt
Output: 
    - performance metrics saved in output file (default: lr_metrics.txt)
"""


# import dependencies 
import os
import sys
sys.path.append(os.path.join("..", ".."))

# utils for classifier
import numpy as np
import utils.classifier_utils as clf_util

# sci-kit learn
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# arparse
import argparse


def main(): 
    
    ### ARUGMENT PARSER ###
    # initialising argument parser, taking input of output filename
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output_filename", help="Name of the output file of metrics", default="lr_metrics.txt")
    args = vars(ap.parse_args())
    output_filename = args["output_filename"]
    
    ### INPUT DATA ###
    print("[INFO] Getting MNIST data...")
    images, labels = fetch_openml('mnist_784', version=1, return_X_y=True)
    
    ### LR CLASSIFIER ###
    print("[INFO] Initialising classifier...")
    lr_classifier = LR_classifier(images, labels)
    images_train, images_test, labels_train, labels_test = lr_classifier.prepare_data(test_size=0.2)
    lr_model = lr_classifier.train_classifier(images_train, labels_train)
    lr_metrics = lr_classifier.evaluate_classifier(lr_model, images_test, labels_test)
    lr_classifier.save_metrics(lr_model, lr_metrics, output_filename)
    
    ### OUTPUT ###
    print(f"[OUTPUT] Perfomance metrics of the lr-classifier:\n{lr_metrics}")        
    print(f"[INFO] All done, file with metrics saved in out/{output_filename}")
        
    
class LR_classifier:
    
    def __init__(self, images, labels): 

        # saving images and labels as input for the classifier
        self.images = images
        self.labels = labels

    def prepare_data(self, test_size):
        """
        Preprocessing data for classifier, turning data into arrays and applying min-max regularisation
        """
        # turn into arrays
        images = np.array(self.images)
        labels = np.array(self.labels)
        # scale data with min-max regularisation
        scaled_images = (images - images.min())/(images.max() - images.min())
        # split into test and train dataset, based on test size input
        images_train, images_test, labels_train, labels_test = train_test_split(
            scaled_images, labels, random_state=9, test_size=test_size)
        return images_train, images_test, labels_train, labels_test

    def train_classifier(self, images_train, labels_train):
        """
        Training logistic-regression classifier 
        """
        clf = LogisticRegression(penalty='none', tol=0.1, solver='saga', multi_class='multinomial').fit(images_train, labels_train)
        return clf

    def evaluate_classifier(self, clf, images_test, labels_test):
        """
        Evaluating performance of logistic-regression classifier based of predictions on the test data 
        """
        predictions = clf.predict(images_test)
        lr_metrics = metrics.classification_report(labels_test, predictions)
        return lr_metrics
    
    def save_metrics(self, lr_classifier, lr_metrics, output_filename):
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
            output_file.write(f"Output for {lr_classifier}:\n\nClassification Metrics:\n{lr_metrics}")
      
 

if __name__=="__main__":
    main()