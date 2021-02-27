#!/usr/bin/env python

"""
For each image in directory: find width, height and number of channels, split image in 4 quadrats, save each split image as jpg, create a csv file containing the filenmae, width, height for the new images.
Parameters:
    path: str <path-to-image-dir>
    output_path: str <path-for-output-file>
Usage:
    basic_image_processinng.py --path <path-to-image-dir>
Example:
    $ python3 basic_image_processing.py --path data/emojis --output_path output/
"""

# import dependencies
import os
import cv2
import glob
import numpy as np
import pandas as pd
import argparse

import sys
sys.path.append(os.path.join(".."))


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
        
    # create empty dataframe to save image info
    df_split_image_info = pd.DataFrame(columns=["filename", "height", "width"])
    
    # loop through images in image path 
    with os.scandir(args["path"]) as images:
    # for each image in the directory 
        for image in images: 
            # if the image is an actual file
            if image.is_file(): 
                
                # save name and path of image 
                image_name, image_path = image.name, image.path 
                # read image
                cv_image = cv2.imread(image.path) 

                # get height, width and n_channels info of image and print info
                height, width, n_channels = cv_image.shape[0:3] # find height, width, n_channels info
                print(f"name: {image_name}, heigth: {height}, width: {width}, n_channels: {n_channels}")

                # split the image into 4 equal parts, based on y-center and x-center
                y_center = int(height/2) 
                x_center = int(width/2)
                
                # split it into four images save image and append info to data frame
                # top left
                top_left = cv_image[0:y_center, 0:x_center]
                top_left_name = image_name[:-4] + "_top_left.jpg"
                cv2.imwrite(os.path.join(output_directory, top_left_name), top_left)
                df_split_image_info = df_split_image_info.append(
                    {"filename": top_left_name, 
                     "height": top_left.shape[0], 
                     "width": top_left.shape[1]}, ignore_index = True)
                
                # top right
                top_right = cv_image[0:y_center, x_center:width]
                top_right_name = image_name[:-4] + "_top_left.jpg"
                cv2.imwrite(os.path.join(output_directory, top_right_name), top_right)
                df_split_image_info = df_split_image_info.append(
                    {"filename": top_right_name, 
                     "height": top_right.shape[0], 
                     "width": top_right.shape[1]}, ignore_index = True)
                
                # bottom_left
                bottom_left = cv_image[y_center:height, 0:x_center]
                bottom_left_name = image_name[:-4] + "_top_left.jpg"
                cv2.imwrite(os.path.join(output_directory, bottom_left_name), bottom_left)
                df_split_image_info = df_split_image_info.append(
                    {"filename": bottom_left_name, 
                     "height": bottom_left.shape[0], 
                     "width": bottom_left.shape[1]}, ignore_index = True)
                
                # bottom_right
                bottom_right = cv_image[y_center:height, x_center:width]
                bottom_right_name = image_name[:-4] + "_top_left.jpg"
                cv2.imwrite(os.path.join(output_directory, bottom_right_name), bottom_right)
                df_split_image_info = df_split_image_info.append(
                    {"filename": bottom_right_name, 
                     "height": bottom_right.shape[0], 
                     "width": bottom_right.shape[1]}, ignore_index = True)
             

    # save csv in output directory
    df_split_image_info.to_csv(os.path.join(output_directory, "df_split_image_info.csv"))

                               

if __name__ == "__main__": 
    main()

                                   
                                   
                                   