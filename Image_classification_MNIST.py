#Import required libraries.
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense ,Flatten ,Conv2D ,MaxPooling2D 

# #### STEP-1:Load and preprocess the data

#Load the MNIST data.
mnist_data = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_data.load_data()

#Scaling the data size between 0 to 1.
def scale_mnist_data(train_images, test_images):
    return train_images/255,test_images/255

scaled_train_images, scaled_test_images = scale_mnist_data(train_images, test_images)

# Add a dummy channel dimension
scaled_train_images = scaled_train_images[..., np.newaxis]
scaled_test_images = scaled_test_images[..., np.newaxis]


# #### STEP-2:Build the convolutional neural network model

#Building the model.
def get_model(input_shape):
    model = Sequential()
    model.add(Conv2D(8, (3,3), padding = 'SAME',activation = 'relu', input_shape = input_shape ))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    return model

model = get_model(scaled_train_images[0].shape)


# #### STEP-3:Compile the model

#Compiling the model.
def compile_model(model):
    model.compile(optimizer = 'adam',
                 loss = 'sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
compile_model(model)


# #### STEP-4:Fit the model to the training data

#Fitting the model.
def train_model(model, scaled_train_images, train_labels):
    history = model.fit(scaled_train_images,train_labels,epochs = 5)
    return history

#Train the model
history = train_model(model, scaled_train_images, train_labels)


# #### STEP-5:Plot the learning curves

frame = pd.DataFrame(history.history)

#Accuracy vs Epochs plot
acc_plot = frame.plot(y="accuracy", title="Accuracy vs Epochs", legend=False)
acc_plot.set(xlabel="Epochs", ylabel="Accuracy")

#Loss vs Epochs plot
acc_plot = frame.plot(y="loss", title = "Loss vs Epochs",legend=False)
acc_plot.set(xlabel="Epochs", ylabel="Loss")


# #### STEP-6:Evaluate the model

#Evaluating the model.
def evaluate_model(model, scaled_test_images, test_labels):
    """
    This function should evaluate the model on the scaled_test_images and test_labels. 
    Your function should return a tuple (test_loss, test_accuracy).
    """
    return model.evaluate(scaled_test_images, test_labels,verbose=2)

test_loss, test_accuracy = evaluate_model(model, scaled_test_images, test_labels)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")


# #### STEP-7:Model predictions

# Extract model prediction
num_test_images = scaled_test_images.shape[0]

random_inx = np.random.choice(num_test_images, 4)
random_test_images = scaled_test_images[random_inx, ...]
random_test_labels = test_labels[random_inx, ...]

predictions = model.predict(random_test_images)

fig, axes = plt.subplots(4, 2, figsize=(16, 12))
fig.subplots_adjust(hspace=0.4, wspace=-0.2)

for i, (prediction, image, label) in enumerate(zip(predictions, random_test_images, random_test_labels)):
    axes[i, 0].imshow(np.squeeze(image))
    axes[i, 0].get_xaxis().set_visible(False)
    axes[i, 0].get_yaxis().set_visible(False)
    axes[i, 0].text(10., -1.5, f'Digit {label}')
    axes[i, 1].bar(np.arange(len(prediction)), prediction)
    axes[i, 1].set_xticks(np.arange(len(prediction)))
    axes[i, 1].set_title(f"Categorical distribution. Model prediction: {np.argmax(prediction)}")
    
plt.show()
