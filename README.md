# UCL IRDM 2016 Assignment 2 - Collaborative Filtering and Recommendation
By Fahad Syed, Avishkar Bhoopchand, Hipolito Iturraspe

The code in this repository can be used to train a Recurrent Neural Network recommendation model on the Yes.com music playlist training dataset or to evaluate a pre-trained model on the test dataset. 

# Requirements
* Operating System: Mac OS X, Windows or Linux
* Language Platform: Python 2.7 or 3.3+ (follow the [instructions](https://www.python.org/) for your platform)
Libraries:
* Tensorflow 0.7+ (follow the [installation instructions](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#download-and-setup))
* Numpy 1.10+ (This is automatically installed when installing Tensorflow)

# Installation
Clone the repository to your local machine using 

    git clone https://github.com/avishkar58/irdm-collab-filtering/

Download the Yes.com playlist dataset from http://www.cs.cornell.edu/~shuochen/lme/data_page.html and extract the contents to your local machine. On a linux machine, this can be done using the command

    tar -xvf dataset.tar.gz

# Running
## Training
To train the model, navigate to the deep-cf subdirectory underneath the path where you cloned the repository. Execute the following command:

    python deepcf.py --mode="training" --data_path="[PATH-TO-DATA]" --model_path="[PATH-TO-SAVE-MODEL]"
  
* [PATH-TO-DATA] refers to the path on your local machine of the train.txt file in the yes_big directory of the required dataset (for example /Users/joebloggs/dataset/yes_big/train.txt) 
* [PATH-TO-SAVE-MODEL] is the path where you want the trained model parameters to be saved (for example ./out/save/mymodel.ckpt)

## Evaluation
To evaluate a pre-trained model, navigate to the deep-cf subdirectory underneath the path where you cloned the repository. Execute the following command:

    python deepcf.py --mode="evaluation" --data_path="[PATH-TO-DATA]" --model_path="[PATH-TO-MODEL]" --train_path="[PATH-TO-TRAIN-DATA]
  
* [PATH-TO-DATA] refers to the path on your local machine of the test.txt file in the yes_big directory of the required dataset (for example /Users/joebloggs/dataset/yes_big/test.txt) 
* [PATH-TO-MODEL] is the path of the saved model parameters (for example ./out/save/mymodel.ckpt)
* [PATH-TO-TRAIN-DATA] is the path to the dataset used to train the model. This is only needed to ensure consistency between the songs in the training and test sets. 


