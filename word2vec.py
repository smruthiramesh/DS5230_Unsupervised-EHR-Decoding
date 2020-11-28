#simple word2vec model training on dataset

import numpy as np
from utils import read_records, adv_tokenizer, encode_labels
from gensim.models import Word2Vec
import scispacy
import en_core_sci_md
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
nlp = en_core_sci_md.load()

# from spacy.lang.en import English
# en = English()

#loading train records
train_records = read_records('train/')
train_text = [train_records[x]['text'] for x in train_records]
train_tags = [train_records[x]['tags'] for x in train_records]

#list of unique criteria
criteria = ['ABDOMINAL', 'ADVANCED-CAD', 'ASP-FOR-MI', 'ENGLISH', 'MAKES-DECISIONS', 'ALCOHOL-ABUSE', 'DIETSUPP-2MOS', 'CREATININE', 'HBA1C', 'MAJOR-DIABETES', 'MI-6MOS', 'DRUG-ABUSE', 'KETO-1YR']

#encoding tags with unique labels 
train_labels = encode_labels(train_tags, criteria)

#tokenizing text using scispacy
train_text_list = []
for text in train_text:
    #only adding valid scispacy entities
    doc = nlp(text)
    train_text_list.append([x.text for x in doc.ents])


#splitting into train and dev sets for testing purposes
X_train, X_dev, y_train, y_dev = train_test_split() 

#training word vectors on dataset
model = Word2Vec(train_text_list, min_count=1,size= 50,workers=3, window =3, sg = 1)

