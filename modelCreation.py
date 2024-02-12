import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.applications import VGG16

# Aggiunto il batch size
batch_size = 32

# Imposta il numero desiderato di epoche
epochs = 5

# Directory principale di addestramento
train_data_dir = r'set/train/train'

# Directory principale di test
test_data_dir = r'set/test'

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 45,
                                   horizontal_flip=True,
                                   shear_range = 0.3,validation_split=0.2,
                                   zoom_range = 0.3)

train_data = train_datagen.flow_from_directory(train_data_dir,
                                              target_size = (100, 100), 
                                              class_mode = 'categorical', batch_size = 64, subset="training")

valid_data = train_datagen.flow_from_directory(train_data_dir,
                                              target_size = (100, 100), 
                                              class_mode = 'categorical', batch_size = 64, subset="validation")

vgg16 = VGG16(input_shape = (100,100, 3), weights = 'imagenet', include_top = False)

for layer in vgg16.layers:
  layer.trainable = False

# Mappa delle classi
class_mapping = train_data.class_indices
print(f'Etichette: {class_mapping}')

# Numero effettivo di classi
num_classes = len(class_mapping)
print(f'Numero classi: {num_classes}')

flatten = Flatten()(vgg16.output)

dense = Dense(256, activation = 'relu')(flatten)
dense = Dropout(0.5)(dense)
dense = Dense(100, activation = 'relu')(dense)
dense = Dropout(0.3)(dense)

# Output Layer
prediction = Dense(33, activation = 'softmax')(dense)

model = Model(inputs = vgg16.input, outputs = prediction)

model.summary()

# Compile the Model
model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer='adam')

# Fit the model
history = model.fit_generator(train_data,validation_data=valid_data,epochs=15)

keras_file="Model.h5"
model.save(keras_file)