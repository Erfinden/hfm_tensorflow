import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the trained model
model = tf.keras.models.load_model('mein_model.h5')

# Define the path to the image you want to classify
image_path = 'C:/Users/htler/Documents/hfm/test5.jpg'

# Load the image
img = Image.open(image_path)
img_array = np.array(img)

# Preprocess the image (normalize and resize)
img_array = img_array / 255.0
img_array = tf.image.resize(img_array, [720, 1280])

# Make the prediction
prediction = model.predict(np.expand_dims(img_array, axis=0))[0][0]

# Print the prediction
if prediction < 0.5:
    print('Das Bild ist leer.')
else:
    print('Das Bild ist voll.')
print('Prediction:', prediction)
