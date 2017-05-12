from flask import Flask, render_template, request
from pymongo import MongoClient
import requests
import pymongo
import json
import os
import pandas as pd
from nltk import word_tokenize
import string


app = Flask(__name__)

client = MongoClient() #instantiate a client that will be connected to Mongo
database = client['foodlogs_database'] #create a variable holding a reference to the db you want to connect to
collection = database['foodlogs_collection']#create a variable to hold the collection you want to connect to
labels = ['meal','time','food','drink','quantity','unit','comment','other']

INDEX = 0
current_record_id = ''

def tokenize(input_string):
    '''takes in a food log string, and returns a tokenized list without punctuation'''
    tokenized = word_tokenize(input_string)
    tokenized_without_punctuation = [word for word in tokenized if word not in string.punctuation]
    return tokenized_without_punctuation

def save_to_db(_id, form, collection=collection):
    labels=request.form.getlist('label')
    update = {'$set':{'labels':labels}}
    current_record_id_dict = {}
    current_record_id_dict['_id'] = _id
    collection.update_one(current_record_id_dict, update)

@app.route("/",methods=['GET'])
def index():
    '''Render a simple splash page.'''
    return render_template('index.html')

@app.route("/label",methods=['GET','POST'])
def label():
    """Render a page containing unlabeled food log retrieved from mongodb, and allows user to select labels & submit. when a post request is made on the form, the labelled words are stored back in the mongodb"""

    #retrieves unlabeled food log from mongodb or csv
    #displays tokenized food log
    #multiple choice options(labels) for each token
    #stores response to mongodb

    global labels
    global current_record_id
    #specify where label is none
    single_document = collection.find_one({'labels': {'$exists': False}})
    single_foodlog = single_document['notes']
    single_foodlog_tokenized = tokenize(single_foodlog)
    text = single_foodlog_tokenized
    current_record_id = single_document['_id']
    return render_template('label.html',text=text,labels=labels)

@app.route("/success",methods=['POST'])
def meow():
    save_to_db(current_record_id, request.form)
    return ''' <a href='/label'>Great job</a>'''

if __name__=="__main__":
    app.run(host='0.0.0.0', debug=True)
