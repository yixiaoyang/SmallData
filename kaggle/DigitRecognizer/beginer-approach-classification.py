#!/usr/bin/python
# -*- coding: utf-8 *-*
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sklearn.svm as svm
from sklearn.model_selection import train_test_split


# loading the data
labeled_images = pd.read_csv("./data/train.csv")

# .iloc[] is primarily integer position based (from 0 to length-1 of the axis), but may also be used with a boolean array.
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]

# Split pandas DataFrame into random train and test subsets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

# normalization
test_images /= 255
train_images /= 255

# svm training
cfunc = svm.SVC()
cfunc.fit(train_images, train_labels.values.ravel())
print cfunc.score(test_images, test_labels)

# label the test images
test_data = pd.read_csv("./data/test.csv")
test_data /= 255
results = cfunc.predict(test_data)

for i in range(8):
    plt.subplot(330 + (i+1))
    img = test_data.iloc[i].as_matrix().reshape((28,28))
    plt.imshow(img, cmap='binary')
    plt.title(results[i])
    # print results[i]
    # img = test_data.iloc[i].as_matrix().reshape((28,28))
    # plt.title(results[i])
    # plt.imshow(img,cmap='binary')
