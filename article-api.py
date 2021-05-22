import requests

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy
import os

import re
import keras
import string

import random

import flask
from flask import request, jsonify
from flask_cors import CORS

from gevent.pywsgi import WSGIServer


#Define articlebot function

def articlebot(subjectinput):
	one_step_model = tf.saved_model.load('use-this-model')

	seedphrase = subjectinput
	tobeword = "is "
	if seedphrase[-1] == "s":
	  tobeword = "are "
	randomtitle = random.randint(1, 3)
	if randomtitle == 1:
	  title = "What " + tobeword + seedphrase + "?"
	if randomtitle == 2:
	  title = "Understanding " + seedphrase
	if randomtitle == 3:
	  title = "Red Hat's approach to " + seedphrase

	initialtext = title + "\n" + seedphrase.capitalize() + " " + tobeword

	#initialtext = 'What are linux computers? \nLinux computers are'

	length = random.randint(3000, 5000)

	class OneStep(tf.keras.Model):
	  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
	    super().__init__()
	    self.temperature = temperature
	    self.model = model
	    self.chars_from_ids = chars_from_ids
	    self.ids_from_chars = ids_from_chars

	    skip_ids = self.ids_from_chars(['', '[UNK]'])[:, None]
	    sparse_mask = tf.SparseTensor(
		values=[-float('inf')]*len(skip_ids),
		indices=skip_ids,
		dense_shape=[len(ids_from_chars.get_vocabulary())])
	    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

	  @tf.function
	  def generate_one_step(self, inputs, states=None):
	    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
	    input_ids = self.ids_from_chars(input_chars).to_tensor()

	    predicted_logits, states = self.model(inputs=input_ids, states=states,
		                                  return_state=True)
	    predicted_logits = predicted_logits[:, -1, :]
	    predicted_logits = predicted_logits/self.temperature
	    predicted_logits = predicted_logits + self.prediction_mask

	    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
	    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

	    predicted_chars = self.chars_from_ids(predicted_ids)

	    return predicted_chars, states

	states = None
	next_char = tf.constant([initialtext])
	result = [next_char]

	for n in range(length):
	  next_char, states = one_step_model.generate_one_step(next_char, states=states)
	  result.append(next_char)

	result = tf.strings.join(result)
	articleout = (result[0].numpy().decode('utf-8'))

	#find last period in article and delete all characters after it
	articleout = articleout[0:(articleout.rfind('.')+1)]

	#change first line of articleout to an H1

	firstline = (articleout.find('\n'))
	if (firstline > 200) :
	  firstline = articleout.find(' ',200)
	 
	articleout = "<h1>" + articleout[:firstline] + "</h1>\n" + articleout[firstline:]

	#change other lines of articleout to p or H2 depending on length

	articleout = articleout.replace("\n\n", "\n")
        
	location = firstline + 10
	lastlocation = location
	lastlinewasheader = True
	i = 0

	while True:
	    
	    location = articleout.find("\n", location)
	    i = i+1
	    if i > 100:
	      break
	    if location == -1: 
	      break
	    linelength = location - lastlocation
	       
	    if (0 <= linelength <= 50) & (lastlinewasheader == False):
	      articleout = articleout[:lastlocation] + "<h2>" + articleout[lastlocation:location] + "</h2>\n" + articleout[location:]
	      lastlinewasheader = True
	      location = location + 11
	    else:
	      articleout = articleout[:lastlocation] + "<p>" + articleout[lastlocation:location] + "</p>\n" + articleout[location:]  
	      lastlinewasheader = False
	      location = location + 9
	    lastlocation = location

	articleout = articleout[:lastlocation] + "<p>" + articleout[lastlocation:(len(articleout))] + "</p>"

	return articleout



#-----------------------------
#Run the API

app = flask.Flask(__name__)
CORS(app)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    if 'subject' in request.args:
        subject = request.args['subject']
        return articlebot(subject)
    else:
        return "<p>Hmm, could not detect an input subject.</p>"

#app.run(host='0.0.0.0', port=5000)
if __name__ == '__main__':
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()





