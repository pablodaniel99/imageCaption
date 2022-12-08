# Import the necessary libraries for building the Image Caption Generator
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, TimeDistributed

# Create a new model with an input layer and an LSTM layer
model = Sequential()
model.add(Input(shape=(img_width, img_height, 3)))
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
    img = load_img(img_path, target_size=(img_width, img_height))
    img = img_to_array(img)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # Use the model to generate a caption for the image
    caption = model.predict(img)[0]

    # Convert the caption from a sequence of words to a sentence
    sentence = " ".join([vocab[i] for i in np.argmax(caption, axis=1)])
    return sentence