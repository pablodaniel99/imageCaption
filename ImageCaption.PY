# Import the necessary libraries for building the Image Caption Generator
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, TimeDistributed

# Load the ResNet50 pre-trained model and remove the output layer
resnet = ResNet50(include_top=False, weights="imagenet")
output = resnet.layers[-1].output
resnet = Model(resnet.input, output)

# Create a new model with the ResNet50 output as input and an LSTM layer
model = Sequential()
model.add(resnet)
model.add(LSTM(256, return_sequences=True))

# Add a TimeDistributed layer to map the LSTM output to a sequence of words
model.add(TimeDistributed(Dense(vocab_size, activation="softmax")))

# Compile the model with a categorical cross-entropy loss and an Adam optimizer
model.compile(loss="categorical_crossentropy", optimizer="adam")

# Train the model on the training data
model.fit(x_train, y_train, epochs=10)

# Test the model on the test data
model.evaluate(x_test, y_test)

# Use the model to generate a caption for an image
def generate_caption(img_path):
    # Load the image and preprocess it
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    img = np.expand_d
