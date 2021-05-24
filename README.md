<h1>Meet Hatbot</h1>

Hatbot accepts any word or phrase then fearlessly tries to write an article in the style of redhat.com. It was trained on a corpus of 200 redhat.com articles using a machine learning algorithm.

Humans interface with Hatbot through a web app (index.html).

The web app calls a python script (article-api.py). This has to be running persistently in the background for anything to work. It uses Flask and Gevent to serve an API.

The script relies on a TensorFlow model (saved in the "use-this-model" folder). I used a different series of scripts in Google Colab to train the model, roughly following [this tutorial](https://www.tensorflow.org/tutorials/text/text_generation). I might update the model from time to time if I can get it working better.

A demo is at [hatbot.site](https://hatbot.site)
