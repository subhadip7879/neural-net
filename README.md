# Study of Convolutional neural networks using Tensorflow to create a house and an image classifier 

## The repository has 3 scripts 
### 1) Basic-neural-net 
this script to create a single layer [simple neural network](http://computing.dcu.ie/~humphrys/Notes/Neural/single.neural.html) using [backpropagation](http://neuralnetworksanddeeplearning.com/chap2.html) via gradient descent to train our network and make our prediction as accurate as possible.
#### Dependecies 
[numpy](http://www.numpy.org/)
### 2) House Classifier 
The folder contains a script to classify a house based on number of bathrooms, price and are as good or bad. The data is tabulated and [pandas](http://pandas.pydata.org/) is the library which helps us in dealing with table like data.After data is tabulated, the features are introduced which allows [Tensorflow](https://www.tensorflow.org/) to begin training process after specifying different like learning rate, training epochs and display step.Since this is a single layer network, the accuracy tends to be around 70% to 80%. the data used for traing and output is also included in the folder. 

#### Dependencies 
- [matplotlib](https://matplotlib.org/
)
- [TensorFlow](ttps://www.tensorflow.org/)
- [pandas](http://pandas.pydata.org/)
- [numpy](http://www.numpy.org/)

### Image Classifier
This is a convolutional neural network which has about 95 - 99% accuracy for [mnist](https://www.tensorflow.org/get_started/mnist/beginners) data set for hand written digits. They work in small filters across the input image and the filters are reused in detecting patterns in the input image because of which the CNNs are quicker to train.The input image is processed in the first convolutional layer using the filter-weights. This results in 16 new images, one for each filter in the convolutional layer.These 16 smaller images are then processed in the second convolutional layer.
