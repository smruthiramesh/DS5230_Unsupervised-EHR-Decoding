from word2vec import word2vec,process_text
import itertools
from clustering import *
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
import scipy
import json
import pickle

criteria = ['ABDOMINAL', 'ADVANCED-CAD', 'ASP-FOR-MI', 'DIETSUPP-2MOS', 'CREATININE', 'MAJOR-DIABETES']
short_forms = ['AB', 'CAD', 'MI', 'DIET', 'CREAT', 'DIAB']
label_mappings = dict(zip(range(len(criteria)),criteria))
short_form_dict = {k:v for k,v in zip(criteria,short_forms)}

train_path = './data/train.txt'
test_path = './data/dev.txt'

#loading best parameters for word2vec clustering
with open('data/best_params_word2vec.pkl','rb') as f:
    best_params = pickle.load(f)

#training word2vec on best params and clustering using gmm
best_embeddings = word2vec(criteria, './data/train.txt','./data/dev.txt',True,best_params[0],best_params[1],best_params[2])
pca = PCA(n_components=2)
pca_X = pca.fit_transform(best_embeddings)
gmm,pca_score_gmm = gmm_clustering(pca_X,n_comp=67,plot=False)
# km,score_km = kmeans_clustering(pca_X,n_comp=67,plot=False)

#finding centers in gmm
train1_indices = np.zeros(67)
for i in range(gmm.n_components):
    density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(pca_X)
    train1_indices[i] = int(np.argmax(density))

#loading train records
with open(train_path,'r') as train:
    train_records = json.load(train)
#loading dev1 records
with open(test_path,'r') as dev:
    dev_records = json.load(dev)
#adding dev1 to train for word embeddings purpose
train_records.update(dev_records)    

train_text = [train_records[x]['text'] for x in train_records]
train_tags = [train_records[x]['tags'] for x in train_records]


train1_indices = [int(i) for i in train1_indices]
#separating train1 and train2
total_indices = set(list(range(len(train_text))))
train2_indices = list(total_indices-set(train1_indices))

#saving indices
with open('data/train1_indices.pkl', 'wb') as f1:
    pickle.dump(train1_indices, f1)
with open('data/train2_indices.pkl', 'wb') as f2:
    pickle.dump(train2_indices, f2)

