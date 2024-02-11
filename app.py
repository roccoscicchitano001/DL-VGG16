from flask import Flask, render_template, request
from tensorflow import keras
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input
import os
from keras.preprocessing import image

app = Flask(__name__)
model='model.h5'
etichette = {
    0: 'Apple Braeburn',
    1: 'Apple Granny Smith',
    2: 'Apricot',
    3: 'Avocado',
    4: 'Banana',
    5: 'Blueberry',
    6: 'Cactus fruit',
    7: 'Cantaloupe',
    8: 'Cherry',
    9: 'Clementine',
    10: 'Corn',
    11: 'Cucumber Ripe',
    12: 'Grape Blue',
    13: 'Kiwi',
    14: 'Lemon',
    15: 'Limes',
    16: 'Mango',
    17: 'Onion White',
    18: 'Orange',
    19: 'Papaya',
    20: 'Passion Fruit',
    21: 'Peach',
    22: 'Pear',
    23: 'Pepper Green',
    24: 'Pepper Red',
    25: 'Pineapple',
    26: 'Plum',
    27: 'Pomegranate',
    28: 'Potato Red',
    29: 'Raspberry',
    30: 'Strawberry',
    31: 'Tomato',
    32: 'Watermelon'
}
target_img = os.path.join(os.getcwd() , 'static/images')

@app.route('/')
def index_view():
    return render_template('index.html')

#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
# Function to load and prepare the image in right shape
def read_image(filename):

    img = load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path)
            class_prediction=model.predict(img)
            # find the index of the class with maximum score
            pred = np.argmax(class_prediction, axis=-1)
            # print the label of the class with maximum score
            fruit=etichette[pred[0]]
            return render_template('predict.html', fruit = fruit,prob=class_prediction, user_image = file_path)
        else:
            return "Unable to read the file. Please check file extension"

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=8000)