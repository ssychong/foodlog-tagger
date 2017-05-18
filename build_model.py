import os
import argparse
import cPickle as pickle
import numpy as np
import pandas as pd
import featurize
from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation

#define an argument parser to take the C value as an input argument. C is a hyperparameter that controls how specific you want the model to be without losing the power to generalize.
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Trains the CRF classifier')
    parser.add_argument("--c-value", dest="c_value", required=False, type=float, default=1.0, help="The C value that will be used for training")
    return parser

#define a class to handle all CRF-related processing.
class CRFTrainer(object):
    #define an init function to initialize the values.
    def __init__(self, c_value, classifier_name='ChainCRF'):
        self.c_value = c_value
        self.classifier_name = classifier_name
        #using chain crf to analyze the data, so add an error check for this:
        if self.classifier_name == 'ChainCRF':
            model = ChainCRF()
        #define the classifier to use with CRF model.
            self.clf = FrankWolfeSSVM(model=model, C=self.c_value, max_iter=100)
        else:
            raise TypeError('Invalid classifier type')

    def load_clean_data(self):
        '''
        load the data into X and y, where X is a numpy array of samples where each sample has the shape (n_letters, n_features)
        '''
        df = featurize.get_data()
        featurize.split_words(df)
        featurize.first_letter_uppercase(df)
        featurize.has_number(df)
        featurize.has_slash(df)
        featurize.spacy_pos_tagger(df)
        featurize.pos_ngrams(df)
        featurize.encoding_labels(df)
        X, y = featurize.get_X_and_y(df)
        return df, X, y

    def train(self, X_train, y_train):
        '''
        training method
        '''
        self.clf.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        '''
        method to evaluate the performance of the model
        '''
        return self.clf.score(X_test,y_test)

    def classify(self, input_data):
        '''
        method to run the classifier on input data
        '''
        return self.clf.predict(input_data)[0]

def decoder(arr):
    '''
    purpose: prediction from model is a numbered array. in order to check the output and make it readable, we need to transform the numbers back into their labels.
    '''
    labels = ['meal','time','food','drink','quantity','unit','comment','other']
    prediction = []
    for i in arr:
        prediction.append(labels[i])
    return prediction

def convert_y_to_tag(y):
    '''
    same as decoder function, but takes all predictions
    '''
    map_inverse = {0:'meal',1:'time',2:'food',3:'drink',4:'quantity',5:'unit',6:'comment',7:'other'}
    predictions = []
    for pred in y:
        predictions.append(np.vectorize(map_inverse.get)(pred))
    return predictions

if __name__ == '__main__':
    #parse the input arguments
    args = build_arg_parser().parse_args()
    c_value = args.c_value
    #initialize the variable with the class and C value:
    crf = CRFTrainer(c_value)
    #load the data:
    df, X, y= crf.load_clean_data()
    #separate data into training and testing datasets:
    X_train, X_test, y_train, y_test, notes_train, notes_test = train_test_split(X, y, df['notes'].values, test_size=0.25)

    #train the CRF model:
    print "\nTraining the CRF model..."
    crf.train(X_train,y_train)

    #evaluate the performance of the CRF model:
    score = crf.evaluate(X_test, y_test)
    print "\nAccuracy score =", str(round(score*100, 2)) + '%'

    #take a random test vector and predict the output using the model:
    print "\nOriginal text =", notes_test[0]
    print "True label =", decoder(y_test[0])
    predicted_output = crf.classify([X_test[0]])
    print "Predicted output =", decoder(predicted_output)

    #train the baseline (Naive Bayes) model:
    print "\nTraining the baseline NB model..."
    nmb = MultinomialNB()
    nmb.fit(np.vstack(X_train), np.hstack(y_train))
    nb_score = nmb.score(np.vstack(X_test),np.hstack(y_test))
    print 'Baseline score =', str(round(nb_score*100, 2)) + '%'
