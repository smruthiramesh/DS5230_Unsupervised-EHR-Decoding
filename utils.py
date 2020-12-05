import xmltodict
import re
import os
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

##################################
#Functions for reading EHR files - Ross Marino
#################

def open_ehr(path):
    with open(path, 'r') as file:
        doc=file.read()
    doc_main = xmltodict.parse(doc)['PatientMatching']
    doc_txt = doc_main['TEXT']
    doc_tags = doc_main['TAGS']
    return doc_txt, doc_tags

def create_tags_dict(xml_tags):
    tag_dict = {}
    for name in xml_tags:
        tag_dict[name] = xml_tags[name]["@met"]
    return tag_dict

def clean_text(xml_txt):
    # remove a lot of white space and non-alphanumeric chars
    new_txt = re.sub("[\s\W]{4,}|\_+|\\n|\\t", " ", xml_txt)
    # clean up residual extra non-alphanumeric chars
    new_txt = re.sub("\W{2,}", " ", new_txt)
    return new_txt

def filter_criteria(all_records):
    crit_out = ['ALCOHOL-ABUSE', 'DRUG-ABUSE', 'ENGLISH','HBA1C','KETO-1YR','MAKES-DECISIONS','MI-6MOS']
    for num, record in all_records.items():
        for crit in crit_out:
            all_records[num]['tags'].pop(crit)
    
    return(all_records)

def read_records(directory):
    records = {}
    for filename in os.listdir(directory):
        if filename.endswith('.xml'):
            id_name = filename[0:3]
            xml_txt, xml_tags = open_ehr(os.path.join(directory,filename))
            tags_dict = create_tags_dict(xml_tags)
            clean_main = clean_text(xml_txt)
            records[id_name] = {'tags':tags_dict,
                            'text':clean_main}
    records = filter_criteria(records)
    return records

##################################
# Functions for removing stop words
#################

def make_bag(txt, stopw):
    """Takes one text doc
        tokenizes words on white space
        makes a bag of words without stop words or numbers
      Returns bag of words"""
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

##################################
# Advanced Tokenizer - Anuj Anand
#################
def adv_tokenizer(doc, model, 
                  replace_entities=False, 
                  remove_stopwords=True, 
                  lowercase=True, 
                  alpha_only=True, 
                  lemma=True):
    """Full tokenizer with flags for processing steps
    replace_entities: If True, replaces with entity type
    stop_words: If False, removes stop words
    lowercase: If True, lowercases all tokens
    alpha_only: If True, removes all non-alpha characters
    lemma: If True, lemmatizes words
    """
    parsed = model(doc)
    # token collector
    tokens = []
    # index pointer
    i = 0
    # entity collector
    ent = ''
    for t in parsed:
        # only need this if we're replacing entities
        if replace_entities:
            # replace URLs
            if t.like_url:
                tokens.append('URL')
                continue
            # if there's entities collected and current token is non-entity
            if (t.ent_iob_=='O')&(ent!=''):
                tokens.append(ent)
                ent = ''
                continue
            elif t.ent_iob_!='O':
                ent = t.ent_type_
                continue
        # only include stop words if stop words==True
        if (t.is_stop) & (remove_stopwords):
            continue
        # only include non-alpha is alpha_only==False
        if (not t.is_alpha)&(alpha_only):
            continue
        if len(t) < 3:
            continue
        if lemma:
            t = t.lemma_
        else:
            t = t.text
        if lowercase:
            t.lower() 
        tokens.append(t)   
    return tokens

##################################
# Criteria Label Encoder - Smruthi Ramesh 
#################

def encode_labels(label_dicts, unique_labels):
    '''given a list of dictionaries of the form {CRITERIA:met/not}, 
    this function returns a list for each dictionary 
    with the unique number associated w each criteria that is met'''
    label_lists = []
    label_numbers = {}
    for x in range(len(unique_labels)):
        label_numbers[unique_labels[x]] = x
    for record in label_dicts:
        record_list = []
        for criteria in record:
            if record[criteria] == 'met':
                record_list.append(label_numbers[criteria])
        label_lists.append(record_list)
    return label_lists
            
