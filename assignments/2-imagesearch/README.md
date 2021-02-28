# Assignment 2: Simple Image Search

## Task

__Creating a simple image search script__

Download the Oxford-17 flowers image data set, available at this link:

https://www.robots.ox.ac.uk/~vgg/data/flowers/17/

Choose one image in your data that you want to be the 'target image'. Write a Python script or Notebook which does the following:
- Use the cv2.compareHist() function to compare the 3D color histogram for your target image to each of the other images in the corpus one-by-one.
- In particular, use chi-square distance method, like we used in class. Round this number to 2 decimal places.
- Save the results from this comparison as a single .csv file, showing the distance between your target image and each of the other images. The .csv file should show the filename for every image in your data except the target and the distance metric between that image and your target. Call your columns: filename, distance.
- Print the filename of the image which is 'closest' to your target image

__General instructions:__
- For this exercise, you can upload either a standalone script OR a Jupyter Notebook
- Save your script as image_search.py OR image_search.ipynb
- If you have external dependencies, you must include a requirements.txt
- You can either upload the script here or push to GitHub and include a link - or both!
- Your code should be clearly documented in a way that allows others to easily follow along
- Similarly, remember to use descriptive variable names! A name like hist is more readable than h.
- The filenames of the saved images should clearly relate to the original image

__Purpose:__

This assignment is designed to test that you have a understanding of:
1. how to make extract features from images based on colour space;
2. how to compare images for similarity based on their colour histogram;
3. how to combine these skills to create an image 'search engine'


## Running the Script

__Cloning the Repository__

To run the script `image_search.py`, it is best to clone this repository to your own machine/server. To do this, you can run the following commands in your terminal:

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

For this assignment I used a dataset of memes, which is not openly available (this is also the reason why the script takes images of with different file-extentions as input, e.g. .jpg, .png, .jpeg). However, I also provided the flowers data, which only needs to be unzipped to be used to run the script. Running the following commands should unzip the file and save the images in a directory called `flowers`.

```bash
# unzip flower data
cd data
unzip flowers.zip
cd ..

```

__Running the script__

If you are at the `cds-visual-nd` directory, I'd recommend moving into `assignments/2-imagesearch/` directory. To run the script, you need to specify the path two parameters:
- -p: path to directory of images 
- -t: name of target image
- -o: directory for output csv file (if the directory does not exist, it will be created)

For instance, run: 
```bash
# move into correct directory
cd assignments/2-imagesearch/

# run the script
python3 image_search.py -p ../../data/flowers/ -t image_0001.jpg -o output/
```

__Output__

The output will saved in the `output/` directory, and will contain a .csv file, showing the filename and distance between the target image and each of the other images. Additionally, the filename and distance of the image which is closest to the target image is printed. 