import pymongo
import pandas as pd
import numpy as np
from nltk import word_tokenize
import string
from collections import deque
from spacy.en import English


def get_data():
    '''
    purpose: returns pandas dataframe of food logs stored in mongodb
    input: none
    output: df of foodlogs
    '''
    client = pymongo.MongoClient()
    database = client.foodlogs_database
    collection = database.foodlogs_collection
    cursor = collection.find()
    df = pd.DataFrame(list(cursor))
    return df

def split_words(df):
    '''
    purpose: tokenize food logs in order to engineer features
    input: dataframe
    output: column in dataframe of tokenized food logs
    '''
    df['tokenized'] = df['notes'].apply(word_tokenize)
    df['tokenized'] = df['tokenized'].apply(lambda x:[word for word in x if word not in string.punctuation])

def first_letter_uppercase(df):
    '''
    purpose: feature indicating whether first letter in a word in food log is uppercase
    input: dataframe
    output: column in dataframe
    '''
    df['first_letter_is_uppercase'] = df['tokenized'].apply(lambda x: [int(word[0].isupper()) for word in x])

def has_number(df):
    '''
    purpose: feature indicating whether word in food log includes a numerical digit
    input: dataframe
    output: column in dataframe
    '''
    df['has_number'] = df['tokenized'].apply(lambda x:[any(char.isdigit() for char in word) for word in x])
    df['has_number'] = df['has_number'].apply(lambda x: [int(ele) for ele in x])

def nltk_pos_tagger(df):
    df['nltk_pos'] = df['tokenized'].apply(lambda x: nltk.pos_tag(x))

def spacy_pos_tagger(df):
    '''
    purpose: features indicating part-of-speech of word in food log
    input: dataframe
    output: column in dataframe
    '''
    #join tokens together for spacy to parse
    df['notes_wo_punct'] = df['tokenized'].apply(lambda x: ' '.join(x))
    #replace slashes with "SLASH", and dashes with "DASH", so that scapy doesnt tokenize on / and -. Same with pm (time)- replace with 00.
    df['notes_hacked'] = df['notes_wo_punct'].apply(lambda x: x.replace('/','SLASH').replace('-','DASH').replace('DASHDASH','--').replace('pm','00'))

    parser = English()
    df['POS'] = df['notes_hacked'].apply(lambda x: parser(x))
    df['POS'] = df['POS'].apply(lambda x: [word.pos_ for word in x])
    df['POS'] = df['POS'].apply(lambda x: [str(word) for word in x])

    pos_tags = ['ADJ','ADP','ADV','CCONJ','DET','INTJ','NOUN','NUM','PART','PRON','PROPN','PUNCT','VERB','X']
    for tag in pos_tags:
        col_name = 'is_'+tag
        df[col_name] = df['POS'].apply(lambda x: [1 if word == tag else 0 for word in x])

def pos_ngrams(df):
    '''
    purpose: feature indicating POS of n (n=1, n=2) words before and after word of interest
    input: dataframe
    output: columns in dataframe
    '''
    df['POS-'] = df['POS'].apply(lambda x: (['BOS'] + x)[:-1])
    df['POS--'] = df['POS-'].apply(lambda x: (['BOS'] + x)[:-1])
    df['POS+'] = df['POS'].apply(lambda x: (x + ['EOS'])[1:])
    df['POS++'] = df['POS+'].apply(lambda x: (x + ['EOS'])[1:])

def has_slash(df):
    '''
    purpose: feature indicating whether word in food log has a slash
    input: dataframe
    output: column in dataframe
    '''
    df['has_slash'] = df['tokenized'].apply(lambda x:["/" in word for word in x])
    df['has_slash'] = df['has_slash'].apply(lambda x:[int(ele) for ele in x])

def encoding_labels(df):
    '''
    purpose: encode labels with integers (starting from 0)
    input: DataFrame
    output: column in dataframe
    '''
    mapping = {'meal':0,'time':1,'food':2,'drink':3,'quantity':4,'unit':5,'comment':6,'other':7}

    df['encoded_labels']=df['labels'].apply(lambda x: map(mapping.get, x))
    df['encoded_labels'] = df['encoded_labels'].apply(lambda x: np.asarray(x))


def get_X_and_y(df):
    '''
    purpose: return X, y needed to build model
    input: dataframe
    output: X,y.

    X is an array of samples, with each sample a numpy array of shape (n_nodes, n_features).
    '''

    feature_array = []
    for f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12,f13,f14,f15,f16,f17 in zip(df['has_number'],df['first_letter_is_uppercase'],df['has_slash'],df['is_ADJ'],df['is_ADP'],df['is_ADV'],df['is_CCONJ'],df['is_DET'],df['is_INTJ'],df['is_NOUN'],df['is_NUM'],df['is_PART'],df['is_PRON'], df['is_PROPN'],df['is_PUNCT'],df['is_VERB'],df['is_X']):
        one_array=np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12,f13,f14,f15,f16,f17]).T
        feature_array.append(one_array)
    X = np.asarray(feature_array)
    y = df['encoded_labels'].values
    return X,y

# def main():
#     df = get_data()
#     split_words(df)
#     first_letter_uppercase(df)
#     has_number(df)
#     has_slash(df)
#     spacy_pos_tagger(df)
#     encoding_labels(df)
#     return df
#     X, y = get_X_and_y(df)
#     return X, y

if __name__ == "__main__":
    # main()
    df = get_data()
    split_words(df)
    first_letter_uppercase(df)
    has_number(df)
    has_slash(df)
    spacy_pos_tagger(df)
    pos_ngrams(df)
    encoding_labels(df)
