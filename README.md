# AI Programming with Python Project

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

This project was part of my submissions for Udacity's AI Programming with Python Nanodegree program. In this project, an image classifier was trained with PyTorch,then converted into a command line application, to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application

# Dependencies
* PyTorch
* ArgParse
* PIL
* Numpy

# Training
To train a new network on a data set with train.py
Basic usage: python train.py data_directory
- Prints out training loss, validation loss, and validation accuracy as the network trains

# Prediction
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint
