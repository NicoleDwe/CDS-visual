#!/usr/bin/env python

"""
Edge / Letter Detection Script

For a given image:
    - draw the ROI on the image
    - crop the image based on the ROI
    - apply canny edge detection
    - draw contours around letters
Input parameters:
    image_path: str <path-to-image>
    roi_points: : x1 y1 x2 y2, four integers defining top-left and bottom-right corner of ROI
    output_path: str <path-for-output-file>
Usage:
    edge_detection.py --image_path <path-to-image> --roi_points x1 y1 x2 y2 --output_path <path-for-output-file>
Example:
    $ python edge_detection.py --image_path ../../data/img/weholdtruths.jpeg --roi_points 100 200 500 800 --output_path output/
Output:
    - image_ROI.jpg: image with ROI in green
    - image_cropped.jpg: image cropped with ROI
    - image_letters.jpg: cropped image with contoured letters

"""

# import dependencies 
import os
import sys
sys.path.append(os.path.join(".."))
import cv2
import numpy as np
import argparse


# main function: for each image in directory compare its colors (rgb) it to the given target image
def main():
    
    #### GET IMAGE PATH AND OUTPUT PATH, READ IMAGE ####
    
    # argparse for input parameters
    ap = argparse.ArgumentParser()
    # parameters
    ap.add_argument("-i", "--image_path", required = True, help = "Path to image")
    ap.add_argument("-r", "--roi_points", required = True, help = "Points of ROI in image", nargs='+')
    ap.add_argument("-o", "--output_path", required = True, help = "Path to output directory")
    # parse arguments
    args = vars(ap.parse_args())
    
    # image path
    image_path = args["image_path"]
    # read image
    image = cv2.imread(image_path)
    # get image name (basename) from path and remove extentionn (splittext)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # output path, if it does not exist, creat it
    output_path = args["output_path"]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    #### DRAW AND CROP ROI ####
    
    ROI_points = args["roi_points"]
    # define the two points based on roi input
    top_left_corner = (int(ROI_points[0]), int(ROI_points[1]))
    bottom_right_corner = (int(ROI_points[2]), int(ROI_points[3]))
    # draw rectangle as ROI on image
    image_ROI = cv2.rectangle(image.copy(), top_left_corner, bottom_right_corner, (0,255,0), (2))
    # save image with ROI
    cv2.imwrite(os.path.join(output_path, f"{image_name}_ROI.jpg"), image_ROI)
    
    #### CROP ROI ####
    
    # crop image based on roi points
    image_cropped = image[top_left_corner[1]:bottom_right_corner[1], top_left_corner[0]:bottom_right_corner[0]]
    # save cropped image
    cv2.imwrite(os.path.join(output_path, f"{image_name}_cropped.jpg"), image_cropped)
    
    #### PROCESS IMAGE ####
    
    # blurring
    image_blurred = cv2.GaussianBlur(image_cropped, (7,7), 0)
    # thresholding to make it only black/white
    (_, image_binary) = cv2.threshold(image_blurred, 115, 255, cv2.THRESH_BINARY)
    # apply canny
    image_canny = cv2.Canny(image_binary, 70, 150)
    
    #### DRAW CONTOURS ####
    
    # finding contours
    (contours, _) = cv2.findContours(image_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # drawing contours on cropped image
    image_letters = cv2.drawContours(image_cropped.copy(), contours, -1, (0,255,0), 2)
    # saving image with contours
    cv2.imwrite(os.path.join(output_path, f"{image_name}_letters.jpg"), image_letters)
    
    print(f"\nDone! Images are saved in {output_path}.")
    
# if the script is called from the terminal, execute main    
if __name__=="__main__":
    main()