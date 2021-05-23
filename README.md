<h1>Meet Hatbot</h1>

Hatbot accepts a word or phrase then tries to write an article about it in the style of redhat.com. It was trained on a corpus of 200 redhat.com articles using a machine learning algorithm. It doesn't do a very good job but it's fun to mess around with.

Humans interface with Hatbot through a web app (index.html).

The web app calls a python script (article-api.py). This has to be running persistently in the background for anything to work. It uses Flask and Gevent to serve an API.

The script relies on a TensorFlow model (saved in the "use-this-model" folder). I used a different series of scripts in Google Colab to train the model, roughly following [this tutorial](https://www.tensorflow.org/tutorials/text/text_generation). I might update the model from time to time if I can get it working better.
