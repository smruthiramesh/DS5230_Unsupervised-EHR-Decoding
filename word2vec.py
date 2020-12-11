#simple word2vec model training on dataset

import numpy as np
from utils import read_records, adv_tokenizer, encode_labels
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score, precision_recall_fscore_support
import json
import matplotlib.pyplot as plt
#loading stopwords
stopwords = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

#tokenizing text using scispacy
def tokenize(doc):
    text_list = []
    for text in doc:
        #only adding valid scispacy entities - lowercase, stemmed, stopwords removed
        tokens = text.split(' ')
        text_list.append([x.lower() for x in tokens if x not in stopwords])
    return text_list

#gets wv representation of a sentence
def sent_to_wv(X):
    X_new = []
    for record in X:
        sent_list = []
        for word in record:
            try:
                sent_list.append(model.wv.get_vector(word))
            except:
                pass            
        sent_array = np.mean(np.array(sent_list), axis=0)
        X_new.append(sent_array)
    return X_new

def process_text(text_dict,criteria)
    train_text = [train_records[x]['text'] for x in text_dict]
    train_tags = [train_records[x]['tags'] for x in text_dict]

    #encoding tags with unique labels 
    train_labels,label_mappings = encode_labels(train_tags, criteria)

    X_train = tokenize(train_text)

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(train_labels)

    return X_train,y_train


def word2vec(criteria, train_file, dev_file):  
    #loading train records
    with open(train_file,'r') as train:
        train_records = json.load(train)
    #loading dev1 records
    with open(dev_file,'r') as dev:
        dev_records = json.load(dev)
    #adding dev1 to train for word embeddings purpose
    train_records.update(dev_records)    
    X_train, y_train = process_text(train_records)
    #training word vectors on dataset
    model = Word2Vec(X_train, min_count=5,size=100,workers=3, window=5, sg = 1)
    #getting representation for train
    X_train_word2vec = np.array(sent_to_wv(X_train))
    return X_train_word2vec


#narrowed down criteria
criteria = ['ABDOMINAL', 'ADVANCED-CAD', 'ASP-FOR-MI', 'DIETSUPP-2MOS', 'CREATININE', 'MAJOR-DIABETES']
embeddings = word2vec(criteria,'./data/train.txt','./data/dev.txt')