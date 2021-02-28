#!/usr/bin/env python

"""
For each image in directory: find width, height and number of channels, split image in 4 quadrats, save each split image as jpg, create a csv file containing the filenmae, width, height for the new images.
Parameters:
    path: str <path-to-image-dir>
    output_path: str <path-for-output-file>
Usage:
    basic_image_processinng.py --path <path-to-image-dir>
Example:
    $ python3 basic_image_processing.py --path ../../data/emojis --output_path output/
"""

# import dependencies
import os
import cv2
import glob
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

import sys
sys.path.append(os.path.join(".."))

# function to read image and get name, height, width, channels
def get_image_info(path_to_image):
    # get string of image part to use with cv2
    str_image_path = str(path_to_image)
    # read image
    image = cv2.imread(str_image_path)
    # get image name
    image_name = os.path.split(path_to_image)[1]
    # get height, width and channel info from image
    height, width, n_channels = image.shape[0:3]
    # return important parameters
    return image, image_name, height, width, n_channels

# define main function 
def main():
    
    # initialise argparse
    ap = argparse.ArgumentParser()
    # define possible arguments
    ap.add_argument("-p", "--path", required=True, help="Path to data/image folder")
    ap.add_argument("-o", "--output_path", required=True, help="Path to output directory")
    # Parse arguments
    args = vars(ap.parse_args())
    
    # if output directory does not exist, create it 
    output_directory = args["output_path"]
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    
    # create output directory for the split images
    split_images_output_directory = os.path.join(output_directory, "split_images")
    if not os.path.exists(split_images_output_directory):
        os.mkdir(split_images_output_directory)
        
        
    #### SPLIT AND SAVE IMAGES ####
    
    # loop through images in image path 
    for image_path in Path(args["path"]).glob("*.png"):
        image, image_name, height, width, n_channels = get_image_info(image_path)[:5]
        print(f"name: {image_name}, heigth: {height}, width: {width}, n_channels: {n_channels}")
        
        # split the image into 4 equal parts, based on y-center and x-center
        y_center = int(height/2) 
        x_center = int(width/2)
                
        # split it into four images save image and append info to data frame
        top_left = image[0:y_center, 0:x_center]
        top_right = image[0:y_center, x_center:width]
        bottom_left = image[y_center:height, 0:x_center]
        bottom_right = image[y_center:height, x_center:width]

        # save the split images with their name
        cv2.imwrite(os.path.join(split_images_output_directory, (image_name[:-4] + "_topleft.jpg")), top_left)
        cv2.imwrite(os.path.join(split_images_output_directory, (image_name[:-4] + "_topright.jpg")), top_right)
        cv2.imwrite(os.path.join(split_images_output_directory, (image_name[:-4] + "_bottomleft.jpg")), bottom_left)
        cv2.imwrite(os.path.join(split_images_output_directory, (image_name[:-4] + "_bottomright.jpg")), bottom_right)
    
    
    #### GET AND SAVE SPLIT IMAGE INFO ####

    # create empty dataframe to save image info
    df_split_image_info = pd.DataFrame(columns=["filename", "height", "width"])
    
    for image_path in Path(split_images_output_directory).glob("*.jpg"):
        image, image_name, height, width, n_channels = get_image_info(image_path)[:5]
        df_split_image_info = df_split_image_info.append(
                    {"filename": image_name, 
                     "height": height, 
                     "width": width}, ignore_index = True)
        
    # save csv in output directory
    df_split_image_info.to_csv(os.path.join(output_directory, "df_split_image_info.csv"))
    

# if called from terminal, execute main
if __name__ == "__main__": 
    main()

                                   
                                   
                    