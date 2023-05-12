import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Definition der Pfade zu den Ordnern mit den Bildern
voll_dir = 'C:/Users/htler/Documents/hfm/bilder/voll'
leer_dir = 'C:/Users/htler/Documents/hfm/bilder/leer'
mittel_dir = 'C:/Users/htler/Documents/hfm/bilder/mittel'

# Liste der Dateinamen in den Ordnern
voll_files = os.listdir(voll_dir)
leer_files = os.listdir(leer_dir)
mittel_files = os.listdir(mittel_dir)

# Leeres Array zum Speichern der Bilddaten
images = []
labels = [] 

# Laden der Bilder aus dem "voll"-Ordner
for file in voll_files:
    img_path = os.path.join(voll_dir, file)
    img = Image.open(img_path)
    img = img.resize((563, 1000))
    img_array = np.array(img)
    images.append(img_array)
    labels.append(2)  # 2 steht für "voll"

# Laden der Bilder aus dem "leer"-Ordner
for file in leer_files:
    img_path = os.path.join(leer_dir, file)
    img = Image.open(img_path)
    img = img.resize((563, 1000))
    img_array = np.array(img)
    images.append(img_array)
    labels.append(0)  # 0 steht für "leer"

# Laden der Bilder aus dem "mittel"-Ordner
for file in mittel_files:
    img_path = os.path.join(mittel_dir, file)
    img = Image.open(img_path)
    img = img.resize((563, 1000))
    img_array = np.array(img)
    images.append(img_array)
    labels.append(1)  # 1 steht für "mittel"

# Konvertierung von images und labels in NumPy-Arrays
images = np.array(images)
labels = np.array(labels)

# Aufteilen der Trainingsdaten in Trainings- und Testdaten
from sklearn.model_selection import train_test_split
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Definition des TensorFlow-Modells
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=train_images.shape[1:]),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax') # Anzahl der Klassen
])

# Kompilieren des Modells
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training des Modells
model.fit(train_images, train_labels, epochs=15, validation_data=(test_images, test_labels))

# Save the model
model.save('mein_model.h5')
