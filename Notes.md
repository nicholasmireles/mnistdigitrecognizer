# Parameter Tweaks
### CNN Model
###### Model filename: CNN.h5
- Removing the pooling layer between the two CNN layers
    - Reduced beginning loss from 14 to 2
- Removing the standardization of the data
    - No change
- Changing the learning rate from .001 to .01
    - No change
- __Getting rid of the image data generator__
    - Huge improvement in speed and accuracy
- Dropout layers
    - Better validation accuracy
- Dropout layer between the dense layers
    - Works if rate < .5
- Dropout layer after CNNs
    - Really effective if < .5
- Increasing epochs:
    - Worse. Overfits quickly.
- Increasing batch size
    - Works up to a point (128) then performance begins to decrease
- Standardizing data by dividing by max value
    - Worse performance. Maxes out ~.98
- Supplementing data with augmentations
    - Overfit very quickly
    - Worth noting that I doubled the test set by randomly rotating the digits between 1 and 90 degrees