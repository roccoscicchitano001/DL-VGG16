import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.applications import VGG16

# Imposta il batch size per il training
batch_size = 32

# Imposta il numero desiderato di epoche per l'addestramento
epochs = 15

# Directory principale dei dati di addestramento
train_data_dir = r'set/train/train'

# Directory principale dei dati di test
test_data_dir = r'set/test'

# Configura il generatore di immagini per il training con diverse trasformazioni
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=45,
                                   horizontal_flip=True,
                                   shear_range=0.3,
                                   validation_split=0.2,
                                   zoom_range=0.3)

# Crea il generatore di dati di addestramento utilizzando il percorso specificato
train_data = train_datagen.flow_from_directory(train_data_dir,
                                              target_size=(100, 100),
                                              class_mode='categorical',
                                              batch_size=64,
                                              subset="training")

# Crea il generatore di dati di validazione utilizzando lo stesso percorso e suddivisione
valid_data = train_datagen.flow_from_directory(train_data_dir,
                                              target_size=(100, 100),
                                              class_mode='categorical',
                                              batch_size=64,
                                              subset="validation")

# Carica il modello VGG16 pre-addestrato e congela i pesi dei suoi strati
vgg16 = VGG16(input_shape=(100, 100, 3), weights='imagenet', include_top=False)
for layer in vgg16.layers:
    layer.trainable = False

# Mappa delle classi ottenuta dal generatore di dati di addestramento
class_mapping = train_data.class_indices
print(f'Etichette: {class_mapping}')

# Numero effettivo di classi
num_classes = len(class_mapping)
print(f'Numero classi: {num_classes}')

# Aggiungi strati personalizzati al modello VGG16
flatten = Flatten()(vgg16.output)
dense = Dense(256, activation='relu')(flatten)
dense = Dropout(0.5)(dense)
dense = Dense(100, activation='relu')(dense)
dense = Dropout(0.3)(dense)

# Strato di output con attivazione softmax per classificazione multiclasse
prediction = Dense(num_classes, activation='softmax')(dense)

# Crea il modello finale utilizzando il modello VGG16 come base
model = Model(inputs=vgg16.input, outputs=prediction)

# Visualizza la struttura del modello
model.summary()

# Compila il modello specificando la funzione di loss, metriche e ottimizzatore
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Addestra il modello utilizzando i generatori di dati di addestramento e validazione
history = model.fit_generator(train_data, validation_data=valid_data, epochs=epochs)

# Salva il modello addestrato
keras_file = "Model.h5"
model.save(keras_file)