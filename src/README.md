# Python Scripts of Visual Analytics Class 

## Simple Search Script: assignment2_nd.py: 
This script is a simple image search script. It compares the 3D color histogram of a target image in a given directory to all other images in the given directory one-by-one. Particularly, the chi-square distance method is used and rounded to 2 decimal places. Results are saved in a .csv file, showing the filename and distance between the target image and each of the other images. Additionally, the filename and distance of the image which is closest to the target image is printed. 

For this assignment the following dataset was used:  https://www.robots.ox.ac.uk/~vgg/data/flowers/17/

To run the script, follow the following steps: 

Clone the repository (saved as cds-visual-nicole):

`git clone https://github.com/nicole-dwenger/cds-visual.git cds-visual-nicole`

Move into the directory cds-visual-nicole:

`cd cds-visual-nicole`

If you also want to use the flowers data, you can unzip the file which is in the data directory. The result should be a directory whithin the data directory called flowers. Then, move back to the parent directory:

`cd data`

`unzip flowers.zip`

`cd ..`

Create the venv and activate it: 

`bash create_vision_venv.sh`

`source cv101/bin/activate`

Run the script, by specifying the required parameters:

- -p: path to directory of images 
- -t: name of target image
- -o: directory for output csv file

Using these parameters and the flowers data the command could be: 

`python3 -p data/flowers/ -t image_0001.jpg -o output/`

