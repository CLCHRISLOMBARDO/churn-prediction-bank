#cluster.py
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import logging
from src.config import PATH_OUTPUT_CLUSTER, SEMILLA
from sklearn.ensemble import RandomForestClassifier

logger=logging.getLogger(__name__)
path_output_cluster=PATH_OUTPUT_CLUSTER

def clustering_kmeans(n_clusters:int , y_train:pd.Series, embedding:np.ndarray ,name:str)->pd.DataFrame:
    name=f"clusters_{n_clusters}"+name
    logger.info(f"Inicio del clustering {name}")
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEMILLA, n_init=10)
    clusters = kmeans.fit_predict(embedding)

    plt.scatter(embedding[:, 0], embedding[:, 1], c=clusters, cmap='viridis')
    plt.colorbar(label='Cluster')

    for cluster_label in sorted(np.unique(clusters)):
        cluster_points = embedding[clusters == cluster_label]
        centroid = cluster_points.mean(axis=0)
        plt.text(centroid[0], centroid[1], str(cluster_label), fontsize=12, ha='center', va='center', color='black',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))

    try:
        file_image=f"{name}.png"
        plt.savefig(path_output_cluster+file_image, dpi=300, bbox_inches="tight")
        logger.info(f"Guardado del cluster {name}")
        logger.info(f"Finalizacion del clustering {name}")
    except Exception as e:
        logger.error(f"Error al guardar el archivo {file_image}")
    plt.close()

    cluster_class_df = pd.DataFrame({'cluster': clusters, 'original_class': y_train})


    return cluster_class_df

    
def cluster_distribution(cluster_class_df:pd.DataFrame  ,prob_baja:np.ndarray )->pd.DataFrame:
    logger.info(f"Comienzo del calculo de la distribucion de bajas y continua en los clusters")
    
    class_distribution_by_cluster = cluster_class_df.groupby('cluster')['original_class'].value_counts().unstack(fill_value=0)
    cluster_prob_df = pd.DataFrame({'cluster': cluster_class_df["cluster"].values, 'prob_baja': prob_baja.flatten()})
    average_prob_baja_by_cluster = cluster_prob_df.groupby('cluster')['prob_baja'].mean()
    class_distribution_by_cluster['average_prob_baja'] = average_prob_baja_by_cluster
    logger.info(f"FIN del calculo de la distribucion de bajas y continua en los clusters")
    return class_distribution_by_cluster

def score_cluster(class_distribution_by_cluster:pd.DataFrame)->int:
    logger.info(f"Comienzo del calculo de vandongenn")
    ct=class_distribution_by_cluster.loc[:, [0, 1]]
    
    n2=2*(sum(ct.apply(sum,axis=1)))
    sumi = sum(ct.apply(np.max,axis=1))
    sumj = sum(ct.apply(np.max,axis=0))
    maxsumi = np.max(ct.apply(sum,axis=1))
    maxsumj = np.max(ct.apply(sum,axis=0))
    vd = (n2 - sumi - sumj)/(n2 - maxsumi - maxsumj)
    logger.info(f"FIN del calculo de vandongenn")
    return vd

def feature_importances_by_cluster(clusters:np.ndarray,X_train_rf)->dict:
        logger.info("Comienzo del feature importance by cluster por agrupamiento")
        important_features_by_cluster = {}
        cluster_series_aligned = pd.Series(clusters, index=X_train_rf.index)
        for cluster in sorted(np.unique(clusters)):
            print(f"Training model for Cluster {cluster} vs. Rest...")
            y_binary = (cluster_series_aligned == cluster).astype(int)

            model = RandomForestClassifier(n_estimators=100, random_state=17, class_weight='balanced') # Added class_weight for imbalanced data
            model.fit(X_train_rf, y_binary)

            importances = model.feature_importances_
            feature_names = X_train_rf.columns

            indices = np.argsort(importances)[::-1]

            important_features_by_cluster[cluster] = [feature_names[i] for i in indices]
            logger.info("FIN del feature importance by cluster por agrupamiento")

        return important_features_by_cluster