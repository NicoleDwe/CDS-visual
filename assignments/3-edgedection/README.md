# Assignment 3: Edge Detection

- [Task](#Task)
- [Running the Script](#Running-the-Script)


## Task

The purpose of this assignment is to use computer vision to extract specific features from images. In particular, we're going to see if we can find text. We are not interested in finding whole words right now; we'll look at how to find whole words in a coming class. For now, we only want to find language-like objects, such as letters and punctuation.

Download and save the image at the link below:

https://upload.wikimedia.org/wikipedia/commons/f/f4/%22We_Hold_These_Truths%22_at_Jefferson_Memorial_IMG_4729.JPG

Using the skills you have learned up to now, do the following tasks:

- Draw a green rectangular box to show a region of interest (ROI) around the main body of text in the middle of the image. Save this as image_with_ROI.jpg.
- Crop the original image to create a new image containing only the ROI in the rectangle. Save this as image_cropped.jpg.
- Using this cropped image, use Canny edge detection to 'find' every letter in the image
- Draw a green contour around each letter in the cropped image. Save this as image_letters.jpg

__TIPS:__
Remember all of the skills you've learned so far and think about how they might be useful
- This means: colour models; cropping; masking; simple and adaptive thresholds; binerization; mean, median, and Gaussian blur.
- Experiment with different approaches until you are able to find as many of the letters and punctuation as possible with the least amount of noise. You might not be able to remove all artifacts - that's okay!

__Bonus challenges:__
If you want to push yourself, try to write a script which runs from the command line and which takes any similar input (an image containing text) and produce a similar output (a new image with contours drawn around every letter).

__General instructions:__
- For this exercise, you can upload either a standalone script OR a Jupyter Notebook
- Save your script as edge_detection.py OR edge_detection.ipynb
- If you have external dependencies, you must include a requirements.txt
- You can either upload the script here or push to GitHub and include a link - or both!
- Your code should be clearly documented in a way that allows others to easily follow along
- Similarly, remember to use descriptive variable names! A name like cropped is more readable than crp.
- The filenames of the saved images should clearly relate to the original image

__Purpose:__

This assignment is designed to test that you have a understanding of:
- how to use a variety of image processing steps;
- how to perform edge detection;
- how to combine these skills in order to find specific features in an image


## Running the Script

__Cloning the Repository__

To run the script `edge_detection.py`, it is best to clone this repository to your own machine/server. To do this, you can run the following commands in your terminal:

```bash
# clone repository into cds-visual-nd
git clone https://github.com/nicole-dwenger/cds-visual.git cds-visual-nd

# move into directory
cd cds-visual-nd
```

__Dependencies__

To run the script, it is best to create the virtual environment cv101 using the bash script `create_venv.sh` which will install the necessary files spectified in the `requirements.txt` file (both files are stored at the root of the directory). To install and activate the environment, you can run the following commands: 

```bash
# create cv101
bash create_vision_venv.sh

# activate cv101 to run the script
source cv101/bin/activate
```

__Data__

For this assignment I used the [image](https://upload.wikimedia.org/wikipedia/commons/f/f4/%22We_Hold_These_Truths%22_at_Jefferson_Memorial_IMG_4729.JPG) which is liked to above and stored in the `data/` directory as weholdtruths.jpeg.

Of course you can also use another image (but I can't promise that the output will be good)

__Running the script__

If you are at the `cds-visual-nd` directory, I'd recommend moving into `assignments/3-edgedetection/` directory. To run the script, you need to specify the path two parameters:
- `-i`: path to the image
- `-r`: points of region of interest, to define area of text, which is used to crop the image
    - This input should be 4 points separated with a space, where the first two numbers represent the x and y values of the top-left-corner of the ROI and the last two numbers the x and y values of the bottom-right-corner of the ROI
    - In the code below, I defined these numbers for the wecountthetruths.jpeg image, if you are providing a different image, you might want to change these numbers.
- `-o`: directory for output files (if the directory does not exist, it will be created)

For instance, run: 
```bash
# move into correct directory
cd assignments/3-edgedetection/

# run the script
python3 edge_detection.py -i ../../data/img/weholdtruths.jpeg -r 1400 890 2900 2800 -o output/
```

__Output__

The output will saved in the `output/` directory, and will contain:
- `"imagename"_ROI.jpg`: The original image, with a green rectangle defining the ROI
- `"imagename"_cropped.jpg`: Only the the ROI of the original image
- `"imagename"_letters.jpg`: The cropped image, with green contours around the letters