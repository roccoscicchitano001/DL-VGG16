# DL-VGG16

**Introduction**


In this blog, we'll see how we can take your work and show it to an audience by deploying your projects on the web. Machine Learning engineers should know the implementation of deployment to use their models on a global scale.
About the Model
I build a model to predict fruit categories. In this model, Have 33 classes. I used VGG trained model to perform this classification problem.

**Saving Trained Models**

You can save your model by calling the save() function on the model and specifying the filename. But usually, we can save the model in 3 formats.

- YAML

- JSON

- HDF5

**Result**

<p>
Epoch 1/3
WARNING:tensorflow:From c:\Users\rscicchitano\Documents\GIT\deep-learning-model-by-flask\venv\lib\site-packages\keras\src\utils\tf_utils.py:492: The name tf.ragged.RaggedTensorValue is deprecated. Please use tf.compat.v1.ragged.RaggedTensorValue instead.

WARNING:tensorflow:From c:\Users\rscicchitano\Documents\GIT\deep-learning-model-by-flask\venv\lib\site-packages\keras\src\engine\base_layer_utils.py:384: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.

527/527 [==============================] - 2544s 5s/step - loss: 0.8164 - accuracy: 0.9361 - val_loss: 1395.1002 - val_accuracy: 0.0152
Epoch 2/3
527/527 [==============================] - 2255s 4s/step - loss: 0.0606 - accuracy: 0.9882 - val_loss: 2302.2778 - val_accuracy: 0.0264
Epoch 3/3
527/527 [==============================] - 2111s 4s/step - loss: 0.0831 - accuracy: 0.9884 - val_loss: 3289.3728 - val_accuracy: 0.0324
Training Completed!
</p>
