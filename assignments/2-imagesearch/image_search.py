#!/usr/bin/env python

"""
For each image in directory compare its rgb distribtution to a specified target image
Parameters:
    path: str <path-to-image-dir>
    target_image: str <name-of-target-img>
    output_path: str <path-for-output-file>
Usage:
    assignment2_nd.py --path <path-to-image-dir>
Example:
    $ python assignment2_nd.py --path data/img --target_image image0001.jpg --output_path output/
"""


# import dependencies 
import os
import sys
sys.path.append(os.path.join(".."))
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import argparse


# function to get files with different file extensions
def get_files(extensions, directory):
    possible_files = []
    for extention in extensions:
        possible_files.extend(Path(directory).glob(extention))
    return possible_files


# main function: for each image in directory compare its colors (rgb) it to the given target image
def main():
    
    # argparse for input parameters
    ap = argparse.ArgumentParser()
    # parameters
    ap.add_argument("-p", "--path", required = True, help = "Path to directory of images")
    ap.add_argument("-t", "--target_image", required = True, help = "Filename of the target image")
    ap.add_argument("-o", "--output_path", required = True, help = "Path to directory for output file")
    # parse arguments
    args = vars(ap.parse_args())
    
    # get path to image directory
    image_directory = args["path"]
    # get name of the target image
    target_name = args["target_image"]
    # get the directory for the output file 
    output_directory = args["output_path"]  
    # if output_directory does not exist, create it 
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    
    # create dataframe to save data
    data = pd.DataFrame(columns=["filename", "distance"])
    
    #### TARGET IMAGE PROCESSING ####
    
    # read target image
    target_image = cv2.imread(os.path.join(image_directory, target_name))
    # create histogram for all 3 color channels
    target_hist = cv2.calcHist([target_image], [0,1,2], None, [8,8,8], [0,256, 0,256, 0,256])
    # normalise the histogram
    target_hist_norm = cv2.normalize(target_hist, target_hist, 0,255, cv2.NORM_MINMAX)
    
    #### COMPARISON IMAGES PROCESSING ####
    
    # define possible extentions
    extentions = ("*.jpg", "*.jpeg", "*.png")
    # use get_files function to get paths for all images with possible extentions
    all_image_paths = get_files(extentions, image_directory)
    
    # for each image path in the list containing all image paths:
    for image_path in all_image_paths:
        # only get the image name from the image_path
        _, image = os.path.split(image_path)
        # if the image is not the target image
        if image != target_name:
            # try processing
            try:
                # read the image and save as comparison image
                comparison_image = cv2.imread(os.path.join(image_directory, image))
                print(image)
                # create histogram for comparison image
                comparison_hist = cv2.calcHist([comparison_image], [0,1,2], None, [8,8,8], [0,256, 0,256, 0,256])
                # normalising the comparison image histogram
                comparison_hist_norm = cv2.normalize(comparison_hist, comparison_hist, 0,255, cv2.NORM_MINMAX)    
                # calculate the chisquare distance
                distance = round(cv2.compareHist(target_hist_norm, comparison_hist_norm, cv2.HISTCMP_CHISQR), 2)
                # append info to dataframe
                data = data.append({"filename": image, 
                                    "distance": distance}, ignore_index = True)
            # if error, pass
            except:
                pass
            
            
    # define output path + filename
    output_path = os.path.join(output_directory, f"{target_name}_comparison.csv")
    # sort values before saving
    data = data.sort_values("distance")
    # save as csv in output directory
    data.to_csv(output_path)
    
    # print that file has been saved
    print(f"output is saved in {output_path} and the closest image to {target_name} is:")
    # print the filename and distance of the image which is closest (min distance)
    print(data[data.distance == data.distance.min()])
    
# if the script is called from the terminal, execute main    
if __name__=="__main__":
    main()