import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('mein_model.h5')

# Load and preprocess the image to be classified
img_path = 'apple.jpg'
img = Image.open(img_path)
img = img.resize((563, 1000))
img_array = np.array(img)
img_array = np.expand_dims(img_array, axis=0) # add a batch dimension
img_array = img_array / 255.0 # normalize pixel values

# Make a prediction
prediction = model.predict(img_array)
rounded_prediction = np.round(prediction, 2)

# Get the predicted class and its confidence level
predicted_class = np.argmax(prediction[0])
confidence = prediction[0][predicted_class]

# Check if model is sure about the classification
range = np.max(prediction[0]) - np.min(prediction[0])
if range < 0.3:
    print("Im not really Sure:")

    # Calculate the percentage of the predicted class
if predicted_class == 0:
    percent = confidence * 50.0
    print('Hackschnitzel ist Leer')
elif predicted_class == 1:
    percent = 25.0 + confidence * 50.0
    print('Hackschnitzel ist Mittel gefuellt')
elif predicted_class == 2:
    percent = 50.0 + confidence * 50.0
    print('Hackschnitzel ist Voll gefuellt')

# Format output string
rounded_prediction_str = ' '.join([f'{x:.2f}' for x in rounded_prediction[0]])
output_str = f'Voll: {rounded_prediction[0][2]:.2f} Mittel: {rounded_prediction[0][1]:.2f} Leer: {rounded_prediction[0][0]:.2f}'

# Print output string and percentage
print(f'{output_str} ({percent:.2f}%)')
