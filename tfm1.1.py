import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Definition des Pfads zum Ordner mit den Bildern im "vollen" Zustand
voll_dir = 'C:/Users/htler/Documents/hfm/bilder/voll'

# Definition des Pfads zum Ordner mit den Bildern im "leeren" Zustand
leer_dir = 'C:/Users/htler/Documents/hfm/bilder/leer'

# Liste der Dateinamen in den beiden Ordnern
voll_files = os.listdir(voll_dir)
leer_files = os.listdir(leer_dir)

# Leeres Array zum Speichern der Bilddaten
images = []
labels = [] 

# Laden der Bilder aus dem "voll"-Ordner
for file in voll_files:
    img_path = os.path.join(voll_dir, file)
    img = Image.open(img_path)
    img_array = np.array(img)
    images.append(img_array)
    labels.append(1)  # 1 steht für "voll"

# Laden der Bilder aus dem "leer"-Ordner
for file in leer_files:
    img_path = os.path.join(leer_dir, file)
    img = Image.open(img_path)
    img_array = np.array(img)
    images.append(img_array)
    labels.append(0)  # 0 steht für "leer"

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
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Kompilieren des Modells
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training des Modells
model.fit(train_images, train_labels, epochs=18, validation_data=(test_images, test_labels))

# Save the model
model.save('mein_model.h5')

import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Definition des Pfads zum Ordner mit den Bildern im "vollen" Zustand
voll_dir = 'C:/Users/htler/Documents/hfm/bilder/voll'

# Definition des Pfads zum Ordner mit den Bildern im "leeren" Zustand
leer_dir = 'C:/Users/htler/Documents/hfm/bilder/leer'

# Liste der Dateinamen in den beiden Ordnern
voll_files = os.listdir(voll_dir)
leer_files = os.listdir(leer_dir)

# Leeres Array zum Speichern der Bilddaten
images = []
labels = [] 

# Laden der Bilder aus dem "voll"-Ordner
for file in voll_files:
    img_path = os.path.join(voll_dir, file)
    img = Image.open(img_path)
    img_array = np.array(img)
    images.append(img_array)
    labels.append(1)  # 1 steht für "voll"

# Laden der Bilder aus dem "leer"-Ordner
for file in leer_files:
    img_path = os.path.join(leer_dir, file)
    img = Image.open(img_path)
    img_array = np.array(img)
    images.append(img_array)
    labels.append(0)  # 0 steht für "leer"

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
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Kompilieren des Modells
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training des Modells
model.fit(train_images, train_labels, epochs=18, validation_data=(test_images, test_labels))

# Save the model
model.save('mein_model.h5')

