from flask import Flask, render_template, request, send_from_directory
from tensorflow import keras
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input, decode_predictions
import os
from keras.preprocessing import image
from keras.applications import VGG16

app = Flask(__name__)
model = VGG16(weights='imagenet')

target_img = os.path.join(os.getcwd() , 'static/images')

@app.route('/')
def index_view():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                          'favicon.ico',mimetype='image/vnd.microsoft.icon')

def label_object(predictions):
    # Decodifica le previsioni in etichette leggibili
    decoded_predictions = decode_predictions(predictions)
    return decoded_predictions[0][0][1]

#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
# Function to load and prepare the image in right shape
def read_image(filename):
    try:
        img = load_img(filename, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x
    except Exception as e:
        print(f"Errore durante la lettura dell'immagine: {e}")
        return None

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path)

            if img is not None:
                class_prediction = model.predict(img)
                # Trova l'indice della classe con il punteggio massimo
                pred = np.argmax(class_prediction, axis=-1)
                object = label_object(class_prediction)
                return render_template('predict.html', object=object, user_image=file_path)
            else:
                return "Errore durante la lettura dell'immagine. Controlla l'estensione del file."
        else:
            return "Impossibile leggere il file. Controlla l'estensione del file."

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=8000)