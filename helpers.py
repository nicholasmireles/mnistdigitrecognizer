import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
img_size = 28
img_shape = (img_size,img_size)
from random import randint

def plot_image(image):
    plt.imshow(image.reshape(img_shape),cmap='binary')
    plt.show()
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

def augment(images):
    temp = []
    for image in images:
        rotation = randint(1,90)
        rotated = ndimage.rotate(image,rotation,reshape=False)
        temp.append(rotated)
    augmented_images = np.stack(temp)
    return augmented_images