# Classification of handwritten digits
# Using ANN (Artificial Neural Network)
# 
# Programmed by: Jean Pierre C. Aldama
# Date: 3/8/2020 5:51 AM
# License: MIT 
# See all NOTE comments for descriptions
# Email: sysad.aldama@gmail.com
# Github: https://www.github.com/sysad-aldama/
# Portal: https://www.quaxiscorp.com/
# Acks: youtube, 

import numpy as np
import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# NOTE Load the dataset
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# NOTE Normalize the dataset to make our ANN easier to train
train_images = (train_images/255) - 0.5
test_images = (test_images/255) - 0.5

# NOTE Flatten the images from 28x28 into 784 dimensional vector (28^2)
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# NOTE Print the shape of training/test images
print(train_images.shape) # Should be 60,000 rows and 784 cols
print(test_images.shape) # Should be 10,000 rows and 784 cols

# NOTE 3 total layers. 2 with 64 neurons each AND the relu function
# NOTE 1 with 10 neurons and the softmax function
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=784))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# NOTE Compile our model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', #(classes that are greater than 2)
    metrics=['accuracy']
)

# NOTE Train our model
model.fit(
    train_images,
    to_categorical(train_labels),  # Ex. 2 it expects [0,0,1,0,0,0,0,0,0,0] 
    epochs=10,                     # More epochs = less loss and more accuracy
    batch_size=32,                 # The number of samples per gradient update
)

# NOTE Evaluate the model
model.evaluate(
    test_images,
    to_categorical(test_labels)
)

# NOTE Save the model
model.save_weights('model.h5')

# NOTE Make predictions using the test_images
predictions = model.predict(test_images[:5])
print(np.argmax(predictions, axis = 1))
print(test_labels[:5])

# NOTE Plot the actual images tested with our model
for i in range(0,5):
    first_image = test_images[i]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28,28))
    plt.imshow(pixels, cmap='gray')
    plt.show()
