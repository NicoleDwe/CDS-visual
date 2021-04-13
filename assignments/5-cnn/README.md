# Assignment 4: Classification using Logistic Regression and Neural Networks

- [Task](#Task)
- [Running the Script](#Running-the-Script)

## Task
__Multi-class classification of impressionist painters__

So far in class, we've been working with 'toy' datasets - handwriting, cats, dogs, and so on. However, this course is on the application of computer vision and deep learning to cultural data. This week, your assignment is to use what you've learned so far to build a classifier which can predict artists from paintings.

You can find the data for the assignment here: https://www.kaggle.com/delayedkarma/impressionist-classifier-data

Using this data, you should build a deep learning model using convolutional neural networks which classify paintings by their respective artists. Why might we want to do this? Well, consider the scenario where we have found a new, never-before-seen painting which is claimed to be the artist Renoir. An accurate predictive model could be useful here for art historians and archivists!

For this assignment, you can use the CNN code we looked at in class, such as the ShallowNet architecture or LeNet. You are also welcome to build your own model, if you dare - I recommend against doing this.

Perhaps the most challenging aspect of this assignment will be to get all of the images into format that can be fed into the CNN model. All of the images are of different shapes and sizes, so the first task will be to resize the images to have them be a uniform (smaller) shape.

You'll also need to think about how to get the images into an array for the model and how to extract 'labels' from filenames for use in the classification report

__Tips__

- You should save visualizations showing loss/accuracy of the model during training; you should also a save the output from the classification report.
- I suggest working in groups for this assignment. The data is quite large and will take some time to move over to worker02. Similarly training the models will be time consuming, so it is preferably to have fewer models on the go.
- You might want to consider a division of labour in your group. One person working on resizing images, one working on extracting labels, one developing the model, etc.
- For reshaping images, I suggest checking out cv.resize() with the cv2.INTER_AREA method
- If you have trouble doing this on your own machines, use worker02.
- Don't worry if the results aren't great! This is a complex dataset we're working with.

__General instructions__

- Save your script as cnn-artists.py
- If you have external dependencies, you must include a requirements.txt
- You can either upload the script here or push to GitHub and include a link - or both!
- Your code should be clearly documented in a way that allows others to easily follow along
- Similarly, remember to use descriptive variable names! A name like X_train is (just) more readable than x1.
- The filenames of the saved images should clearly relate to the original image

__Purpose__

This assignment is designed to test that you have a understanding of:
- how to build and train deep convolutional neural networks;
- how to preprocess and prepare image data for use in these models;
- how to work with complex, cultural image data, rather than toy datasets

---

## Running the Script

__Cloning the Repository__

To run the scripts `cnn-artists-py`, it is best to clone this repository to your own machine/server by running the following commands:

```bash
# clone repository into cds-visual-nd
git clone https://github.com/nicole-dwenger/cds-visual.git cds-visual-nd

# move into directory
cd cds-visual-nd
```

__Dependencies__

To run the script, it is best to create the virtual environment `cv101` using the bash script `create_venv.sh`. This will install the necessary dependencies spectified in the `requirements.txt` file (both files are stored at the root of the directory). To install and activate the environment, you can run the following commands: 

```bash
# create cv101
bash create_vision_venv.sh

# activate cv101 to run the script
source cv101/bin/activate
```

__Data__

For this assignment this dataset of impressionist paintings was used. I provided a sample of the data in the `cds-visual-nd/data/impressionist_subset/` direcotry. If you would like to run the script on the full data I recommend that you download it from Kaggle. 

__Running the script__

If you are at the `cds-visual-nd` directory, I'd recommend moving into `assignments/5-cnn` directory. This is necessary to also ensure that utils from the `utils` directory (sitting at the root directory) can be correctly accessed. To run the script it is not *necessary* to specifc any parameters. This means, to run the script, you can run: 

```bash
# move into correct directory
cd assignments/5-cnn/

# run the script
python3 cnn-artists.py
```

*Optional Parameters:*

The script has default parameters specified for the paths to the training and test data (sample of impressionist data, as specified above), the number of epochs (20) and the batch size (32) of the LetNet model. If you wish to change these you can use the folowing parameters:


| short name | long name     | default |
|------------|---------------|---------|
| -tr        | --train_data  |   ../../data/impressionist_subset/training/training/     |
| -te        | --test_data   |   ../../data/impressionist_subset/validation/validation/   |
| -e         | --n_epochs    |    20   |
| -b         | --batch_size  |    32   |
 

__Output__

The following information will be printed to the console:
- dimension according to which all images are resized, which is based on the minimum width/height of all images
-

The following files will be safed in `out/`
- LeNet_model.png: visualisation of the LeNet model structure
- model_summary.txt: summary of the LeNet model structure
- model_history.png: visualisation of loss and accuracy
- classification_report.txt: classification report
