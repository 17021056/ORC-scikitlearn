import os
import numpy as np
from skimage import io
def get_images(img_file, number):
    f = open(img_file, "rb") # Open file in binary mode
    f.read(16) # Skip 16 bytes header
    images = []

    for i in range(number):
        image = []
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)
        print(image)
    return images

def get_labels(label_file, number):
    l = open(label_file, "rb") # Open file in binary mode

    l.read(8) # Skip 8 bytes header

    labels = []
    for i in range(number):
        labels.append(ord(l.read(1)))
    print(labels)
    np.savetxt("labels.txt", labels)
    return labels

def convert_png(images, labels, directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

    for i in range(len(images)):
        out = os.path.join(directory, "%06d-num%d.png"%(i,labels[i]))
        io.imsave(out, np.array(images[i]).reshape(28,28))

number = 10000
train_images = get_images("mnist/train-images.idx3-ubyte" , number)
train_labels = get_labels("mnist/train-labels.idx1-ubyte", number)

convert_png(train_images, train_labels, "X_train")