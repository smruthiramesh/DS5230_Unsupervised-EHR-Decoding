from utils import read_records
import re
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def make_bag(txt, stopw):
    # clean_txt? or do on upload?
    bow = re.split('\s',txt.lower())
    new_bow=[]
    for word in bow:
        if word not in stopw and len(word)>0 and not re.search('\d',word):
            new_bow.append(word)
    return(new_bow)

def remove_stop_words(raw_corpus, doc_freq=0.75):
    """ Takes in a list of all raw text
    Returns list of raw text without stop words based on document frequency and TFIDF"""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([doc.lower() for doc in raw_corpus])
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    words_tfidf = pd.DataFrame(denselist, columns=feature_names)
    
    new_stopwords = dict.fromkeys(feature_names, 0)
    for (word, data) in words_tfidf.iteritems():
        for num in data.values:
            if num > 0:
                new_stopwords[word] +=1

    new_sw = []
    for word, count in new_stopwords.items():
        if count > doc_freq*len(raw_corpus):
            new_sw.append(word)
    stopw = stopwords.words('english')
    stopw = [*stopw, *new_sw]
    text_nostop = []
    for doc in raw_corpus:
        doc_bag = make_bag(doc, stopw)
        text_nostop.append(" ".join(doc_bag))  
    return(text_nostop)