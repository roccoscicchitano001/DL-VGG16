import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.applications import VGG16

# Aggiunto il batch size
batch_size = 32

# Imposta il numero desiderato di epoche
epochs = 3

# Directory principale di addestramento
train_data_dir = r'set/train/train'

# Directory principale di test
test_data_dir = r'set/test'

# Generatore di dati per addestramento
train_data_gen = ImageDataGenerator().flow_from_directory(
    directory=train_data_dir,
    target_size=(224, 224),
    class_mode='categorical',
    shuffle=True  # Considera di mescolare le immagini durante l'addestramento
)

# Generatore di dati per test
test_data_gen = ImageDataGenerator().flow_from_directory(
    directory=test_data_dir, 
    target_size=(224,224),
    class_mode='categorical',
    shuffle=True  # Considera di mescolare le immagini durante il test
)

# Mappa delle classi
class_mapping = train_data_gen.class_indices
print(f'Etichette: {class_mapping}')

# Numero effettivo di classi
num_classes = len(class_mapping)
print(f'Numero classi: {num_classes}')

# Modello VGG16
vgg = VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False

# Costruzione del modello personalizzato
x = Flatten()(vgg.output)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

training_steps_per_epoch = np.ceil(train_data_gen.samples / batch_size)
validation_steps_per_epoch = np.ceil(test_data_gen.samples / batch_size)
model.fit(train_data_gen, steps_per_epoch = training_steps_per_epoch, validation_data=test_data_gen, validation_steps=validation_steps_per_epoch,epochs=epochs, verbose=1)
print('Training Completed!')

Y_pred = model.predict(test_data_gen, test_data_gen.samples / batch_size)
val_preds = np.argmax(Y_pred, axis=1)
val_trues =test_data_gen.classes
print(classification_report(val_trues, val_preds))

# Salvataggio del modello
keras_file = 'Model.h5'
model.save(keras_file)
