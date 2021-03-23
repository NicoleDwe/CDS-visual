# Assignment 4: Classification using Logistic Regression and Neural Networks

- [Task](#Task)
- [Running the Script](#Running-the-Script)

## Task
__Classifier benchmarks using Logistic Regression and a Neural Network__

This assignment builds on the work we did in class and from session 6.

You'll use your new knowledge and skills to create two command-line tools which can be used to perform a simple classification task on the MNIST data and print the output to the terminal. These scripts can then be used to provide easy-to-understand benchmark scores for evaluating these models.

You should create two Python scripts. One takes the full MNIST data set, trains a Logistic Regression Classifier, and prints the evaluation metrics to the terminal. The other should take the full MNIST dataset, train a neural network classifier, and print the evaluation metrics to the terminal.


__Tips:__
- I suggest using scikit-learn for the Logistic Regression Classifier
- In class, we only looked at a small sample of MNIST data. I suggest using fetch_openml() to get the full dataset, like we did in session 6
- You can use the NeuralNetwork() class that I introduced you to during the code along session
- I recommend saving your .py scripts in a folder called src﻿; and have your NeuralNetwork class in a folder called utils, like we have on worker02
- You may need to do some data manipulation to get the MNIST data into a usable format for your models
- If you have trouble doing this on your own machine, use worker02!

__Bonus Challenges:__
- Have the scripts save the classifier reports in a folder called out, as well as printing them to screen. Add the user should be able to define the file name as a command line argument (easier)
- Allow users to define the number and size of the hidden layers using command line arguments (intermediate)
- Allow the user to define Logistic Regression parameters using command line arguments (intermediate)
- Add an additional step where you import some unseen image, process it, and use the trained model to predict it's value - like we did in session 6 (intermediate)
- Add a new method to the Neural Network class which will allow you to save your trained model for future use (advanced)

__General instructions:__
- Save your script as lr-mnist.py and nn-mnist.py
- If you have external dependencies, you must include a requirements.txt
- You can either upload the script here or push to GitHub and include a link - or both!
- Your code should be clearly documented in a way that allows others to easily follow along
- Similarly, remember to use descriptive variable names! A name like X_train is (just) more readable than x1.
- The filenames of the saved images should clearly relate to the original image

__Purpose:__

This assignment is designed to test that you have a understanding of:
- how to train classification models using machine learning and neural networks;
- how to create simple models that can be used as statistical benchmarks;
- how to do this using scripts which can be executed from the command line

---

## Running the Script

__Cloning the Repository__

To run the script `edge_detection.py`, it is best to clone this repository to your own machine/server by running the following commands:

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

For this assignment the mnist dataset is used. This is downloaded through the script, so there is no additional data needed. 

__Running the script__

If you are at the `cds-visual-nd` directory, I'd recommend moving into `assignments/4-cclassification/` directory. This is necessary to also ensure that utils from the `utils` directory (sitting at the root directory) can be correctly accessed. To run the script it is not *necessary* to specifc any parameters. This means, to run the logistic regression classifier or neural network classifier script, you can run: 

```bash
# move into correct directory
cd assignments/4-classification/

# run the logistic regression classifier script
python3 lr-mnist.py
# run the neural network classifier script
python3 nn-mnist.py
```

*Optional Parameters:*

Both scripts have a default output-filename specified. This means the performance metrics are saved in a .txt file in the `out/` directory as `lr_metrics.txt` or `nn_metrics.txt`. If you wish to change this, you can specify your own name using the `-output-filename` argument, e.g. run: 

```bash
# run the logistic regression classifier script
python3 lr-mnist.py --output_filename my_output_name.txt
```

__Output__

The performance metrics will both be shown on the command line and will saved in the `out/` directory, using either the default filename (`lr_metrics.txt` or `nn_metrics.txt`) or a specified name (see above).

__Notes on the Models__

It should be noted that to reduce the time of running the script I specified the layers of the neural network to be 784-32-16-10. Futher, I reduced the epochs to 10. As the model with these parameters already performed better than the logistic regression, I chose to keep them, but for more reliability they could be changed to more appropriate values. 