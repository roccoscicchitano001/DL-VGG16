# DL-VGG16

**Introduction**


In this blog, we'll see how we can take your work and show it to an audience by deploying your projects on the web. Machine Learning engineers should know the implementation of deployment to use their models on a global scale.
About the Model
I build a model to predict fruit categories. In this model, Have 33 classes. I used VGG trained model to perform this classification problem.

**Saving Trained Models**

You can save your model by calling the save() function on the model and specifying the filename. But usually, we can save the model in 3 formats.

- YAML

- JSON

. HDF5

**Result**

<p>
Il dato finale "527/527 [==============================] - ETA: 0s - loss: 0.8589 - accuracy: 0.9463" indica che hai completato con successo tutte le epoche di addestramento. Ecco come interpretare questi ultimi valori:

Loss (Perdita): L'ultima loss è 0.8589, che è un valore abbastanza basso. Indica quanto il modello si discosta dagli obiettivi desiderati alla fine dell'addestramento. Una loss bassa suggerisce che il modello ha imparato a fare predizioni accurate sui dati di addestramento.

Accuracy (Precisione): L'ultima accuracy è 0.9463, che corrisponde al 94.63%. Questo indica che circa il 94.63% delle predizioni del modello sono corrette sui dati di addestramento.

Questi sono risultati eccellenti e suggeriscono che il tuo modello ha imparato molto bene dai dati di addestramento. Ora puoi considerare di valutare le prestazioni del modello su un set di dati di test separato per assicurarti che il modello generalizzi bene su nuovi dati.
</p>
