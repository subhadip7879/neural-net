# Study of Convolutional neural networks using Tensorflow to create a house and an image classifier 

## The repository has 3 scripts 
### 1) Basic-neural-net 
This script to create a single layer [simple neural network](http://computing.dcu.ie/~humphrys/Notes/Neural/single.neural.html) using [backpropagation](http://neuralnetworksanddeeplearning.com/chap2.html) via gradient descent to train our network and make our prediction as accurate as possible.
#### Dependecies 
[numpy](http://www.numpy.org/)
### 2) House Classifier 
The folder contains a script to classify a house based on number of bathrooms, price and are as good or bad. The data is tabulated and [pandas](http://pandas.pydata.org/) is the library which helps us in dealing with table like data.After data is tabulated, the features are introduced which allows [Tensorflow](https://www.tensorflow.org/) to begin training process after specifying different like learning rate, training epochs and display step.Since this is a single layer network, the accuracy tends to be around 70% to 80%. the data used for traing and output is also included in the folder. 

#### Dependencies 
- [matplotlib](https://matplotlib.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [pandas](http://pandas.pydata.org/)
- [numpy](http://www.numpy.org/)

### 3) Image Classifier
This is a convolutional neural network which has about 95 - 99% accuracy for [mnist](https://www.tensorflow.org/get_started/mnist/beginners) data set for hand written digits. They work in small filters across the input image and the filters are reused in detecting patterns in the input image because of which the CNNs are quicker to train.The input image is processed in the first convolutional layer using the filter-weights. 

This results in 16 new images, one for each filter in the convolutional layer.These 16 smaller images are then processed in the second convolutional layer.We need filter-weights for each of these 16 channels, and we need filter-weights for each output channel of this layer. 

These are then flattened to a single vector of length 7 x 7 x 36 = 1764, which is used as the input to a fully-connected layer with 128 neurons (or elements). This feeds into another fully-connected layer with 10 neurons, one for each of the classes, which is used to determine the class of the image.

The computation in TensorFlow is actually done on a batch of images instead of a single image, which makes the computation more efficient. This means the flowchart actually has one more data-dimension when implemented in TensorFlow.

## The MNIST dataset

![](http://i.imgur.com/TaJkAHl.png)

### Plotting a few omages to see if the data is correct

![](http://i.imgur.com/FqiAFsN.png?1)

### Performance and accuracy after 100 optimization iterations of training CNN

![](http://i.imgur.com/4HSBgln.png?1)

### Performance and accuracy after 1000 optimization iterations of training CNN

![](http://i.imgur.com/PCATzau.png?1)

### Performance and accuracy after 10000 optimization iterations of training CNN

![](http://i.imgur.com/W16XnRx.png?1)

Suggestions/edits are warmly welcome, and can be given by creating an issue/pull request in this repository.