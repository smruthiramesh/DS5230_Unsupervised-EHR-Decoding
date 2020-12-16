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
train_path = './data/train.txt'
test_path = './data/dev.txt'

windows = [3,5,7,10]
sizes = [100,200,300,400]
min_counts = [5,10,15,20]

param_combos = list(itertools.product(*[windows,sizes,min_counts]))

# tsne_scores = []
pca_scores = []

embeddings_1 = word2vec(criteria,train_path,test_path,True)

pca_1 = PCA(n_components=5)
new_X = pca_1.fit_transform(embeddings_1)
tsne_x = get_tsne_features(new_X)
gmm_clustering(tsne_x,n_comp=67,plot=False)
gmm_clustering(new_X,n_comp=67,plot=False)
kmeans_clustering(new_X,plot=False)

for params in param_combos:
    window = params[0]
    size = params[1]
    min_count = params[2]
    embeddings = word2vec(criteria,train_path,test_path,True,window,size,min_count)
    pca = PCA(n_components=2)
    pca_X = pca.fit_transform(embeddings)
    tsne_X = get_tsne_features(pca_X)
    clustering, score = kmeans_clustering(tsne_X,n_comp=67,plot=False)
    pca_scores.append(score)

best_params = param_combos[np.argmax(pca_scores)]

with open('data/best_params_word2vec.pkl', 'wb') as f:
    pickle.dump(best_params, f)