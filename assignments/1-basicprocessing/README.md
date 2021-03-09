# Assignment 1: Basic Image Processing

- [Task](#Task)
- [Running the Script](#Running-the-Script)

## Task

__Basic scripting with Python__

Create or find small dataset of images, using an online data source such as Kaggle. At the very least, your dataset should contain no fewer than 10 images.

Write a Python script which does the following:
- For each image, find the width, height, and number of channels
- For each image, split image into four equal-sized quadrants (i.e. top-left, top-right, bottom-left, bottom-right)
- Save each of the split images in JPG format
- Create and save a file containing the filename, width, height for all of the new images.

__General instructions:__
- For this exercise, you can upload either a standalone script OR a Jupyter Notebook
- Save your script as basic_image_processing.py OR basic_image_processing.ipynb
- If you have external dependencies, you must include a requirements.txt
- You can either upload the script here or push to GitHub and include a link - or both!
- Your code should be clearly documented in a way that allows others to easily follow the structure of your script.
- Similarly, remember to use descriptive variable names! A name like width is more readable than w.
- The filenames of the split images should clearly relate to the original image.

__Purpose:__
This assignment is designed to test that you have a understanding of:
1. how to structure, document, and share a Python script;
2. how to effectively make use of basic functions in cv2;
3. how to read, write, and process images files.


## Running the Script

__Cloning the Repository__

To run the script `basic_image_processing.py`, it is best to clone this repository to your own machine/server. To do this, you can run the following commands in your terminal:

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

# activate cv101 to run the scripts
source cv101/bin/activate
```

__Data__

For this assignment I used a dataset of emojis, downloaded from [Kaggle](https://www.kaggle.com/victorhz/emoticon). The data is stored in the `cds-visual-nd/data/emoji/` directory.

__Running the script__

If you are still at the `cds-visual-nd` directory, I'd recommend moving into `assignments/1-basicprocessing/` directory to run the script. To run the script, you need to specify the path two parameters:
- `-p`: path to the directory containing the input images
- `-o`: path to the directory where output files should be saved (if the directory does not exist, it will be created)

For instance, run: 
```bash
# move into correct directory
cd assignments/1-basicprocessing/

# run the script
python3 basic_image_processing.py -p ../../data/emojis/ -o output/
```

__Output__

The output will saved in the `output/` directory, and will contain the split images, which are named with the name of the original image and the image part, e.g. happyemoji_topleft.jpg, and saved in `output/split_images/`, and a .csv file containing the height and width information of each of the split images. 