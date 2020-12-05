import xmltodict
import re
import os


#Functions for reading EHR files - Ross Marino
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
    return new_txt.lower()

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
    return records

# Advanced Tokenizer - Anuj Anand
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
            
