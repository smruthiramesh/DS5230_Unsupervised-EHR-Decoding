
import numpy as np
import matplotlib.pyplot as plt

from utils import get_tsne_features

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

cluster_colors = {
  0: "crimson",
  1: "slateblue",
  2: "slategrey",
  3: "coral",
  4: "violet",
  5: "yellow"
  }

##################################
# K-Means
##################################
def kmeans_clustering(vectors: np.ndarray, n_comp:int=6,
                      plot=False, fig_size=(10,8)):

  """
  Obtain Kmeans model.
  To plot, set plot=True.
  """

  tsne_features = get_tsne_features(vectors)

  kmeans = KMeans(
    init="random",
    n_clusters=n_comp,
    n_init=1,
    max_iter=300,
    random_state=0
    )

  kmeans.fit(tsne_features)
  labels = kmeans.labels_
  sil_score = silhouette_score(tsne_features, kmeans.labels_).round(2)

  if plot:
    plt.figure(figsize=fig_size)
    km_colors = [cluster_colors[label] for label in labels]
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=km_colors)
    plt.title(
        f"k-means: Silhouette: {sil_score}", fontdict={"fontsize": 12}
    )

  return kmeans


##################################
# GMM
##################################

def gmm_clustering(vectors: np.ndarray, covariance_type:str='full',
                   n_comp=6, plot=False, fig_size=(10,8)):
  """
  Obtain GMM clustering.
  To plot, set plot=True.
  """

  tsne_features = get_tsne_features(vectors)

  # training gaussian mixture model
  gmm = GaussianMixture(n_components=n_comp,
                        covariance_type=covariance_type,
                        random_state=0)

  gmm.fit(tsne_features)
  labels = gmm.predict(tsne_features)
  sil_score = silhouette_score(tsne_features, labels).round(2)

  if plot:
    plt.figure(figsize=fig_size)
    gmm_colors = [cluster_colors[label] for label in labels]
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=gmm_colors)
    plt.title(
        f"k-means: Silhouette: {sil_score}", fontdict={"fontsize": 12}
    )

  return gmm