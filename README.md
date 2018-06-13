# Keras TensorFlow Digit Recognizer
##### v1.5
###### Created by Nicholas Mireles 5/29/2018
This is my basic attempt at creating a deep-learning model for the recognition of hand-written digits from the canonical MNIST dataset.

Everything is hard-coded. This is elegant.


### Current Accuracy
Validation: .9917

Test: .9986

### Changelog
###### 6/13/2018
- Created a CNN model based on a tutorial by Magnus Pederson (https://www.youtube.com/watch?v=HMcx-zY8JSg)
- Removed the image generator as it reduced performance and speed drastically
- Started a notes file to better track what I learn

###### 5/30/2018
- Changed the activation functions to relu
- Added batch normalization between the layers
- Did zero-centering on training and testing sets before adding to network
- Reduced training epochs to 1, since training/validation loss begin diverging after
    - Maybe train MORE epochs to see if they re-converge?