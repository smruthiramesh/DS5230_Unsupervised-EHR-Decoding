
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
def kmeans_clustering(vectors: np.ndarray, method=None, n_comp:int=6,
                      plot=True, fig_size=(10,8), save_fig=False):

  """
  Obtain Kmeans clustering.
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
        f"K-Means + {type}", fontdict={"fontsize": 12}
    )
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    if save_fig:
      plt.savefig("kmeans_"+method+".jpg")

  return (kmeans, sil_score)


##################################
# GMM
##################################

def gmm_clustering(vectors: np.ndarray, method=None, covariance_type:str='full',
                   n_comp=6, plot=True, fig_size=(10,8), save_fig=False):
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
        f"GMM+{type}", fontdict={"fontsize": 12}
    )
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    if save_fig:
      plt.savefig("gmm_"+method+".jpg")

  return (gmm, sil_score)