import torch
import numpy as np
from typing import List

from utils import adv_tokenizer

from allennlp.commands.elmo import ElmoEmbedder
from transformers import BertModel, BertTokenizer

##########################  ELMo  ##########################
# Check for GPU
if torch.cuda.is_available():
  elmo = ElmoEmbedder(cuda_device=0)
else:
  print("GPU not found!!")
  elmo = ElmoEmbedder(cuda_device=-1)


def elmo_embedding_doc_level(ehr_records: List[str])-> np.ndarray:
    # Tokenize each document
    token_docs = [adv_tokenizer(d) for d in ehr_records]

    # Get ELMo embeddings of each word in each tokenized document
    # [-1] -> To get the last set of ELMo embeddings
    elmo_vecs = [elmo.embed_sentence(doc)[-1] for doc in token_docs]

    # Take the mean of each dimension (column) which gives the doc-level average
    doc_level_embedding = np.array([word_vectors.mean(axis=0) for word_vectors in elmo_vecs])

    print("Shape of ELMo vectors:", doc_level_embedding.shape)
    return doc_level_embedding


##########################  BioBERT  ##########################
# BioBERT model
MODEL_NAME = 'dmis-lab/biobert-base-cased-v1.1'

# Load pre-trained model
model = BertModel.from_pretrained(MODEL_NAME)
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


def biobert_embedding_doc_level(ehr_records:List[str],
                                max_len:int=128, win_size:int=5)->np.ndarray:

  embedding_len = 768
  doc_level_embedding = np.zeros((len(ehr_records), embedding_len))
  for idx, text in enumerate(ehr_records):
    # Split text into list of words
    word_list = text.strip().split(' ')
    # Split list of words into chunks of length=max_len
    # Add sliding window concept
    split_word_list = [word_list[i:i + max_len] for i in range(0, len(word_list), max_len-win_size)]

    tokens = tokenizer(split_word_list, is_split_into_words=True,
                      padding=True, return_tensors="pt")

    if torch.cuda.is_available():
      device = torch.device("cuda:0")
      tokens = tokens.to(device)
      model.to(device)

    outputs = model(**tokens)

    # Get last hidden state of model
    last_hidden_state = outputs.last_hidden_state.detach()
    # representation of the 'CLS' token (chunk-level embedding)
    cls_embedding = last_hidden_state[:,0,:].cpu().numpy()
    # Take mean of the CLS tokens
    doc_level_embedding[idx] = np.mean(cls_embedding, axis=0)

  print("Shape of BioBERT vectors: ", doc_level_embedding.shape)
  return doc_level_embedding