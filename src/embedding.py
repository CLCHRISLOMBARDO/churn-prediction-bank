#embedding.py
import pandas as pd
import numpy as np
from umap import UMAP
import logging
from src.config import SEMILLA


logger=logging.getLogger(__name__)
def embedding_umap(md:np.ndarray, n_componentes:int=2,n_neighbors:int=20,min_dist:float=0.77 ,
             learning_rate:float=0.05 , metric:str="precomputed") ->np.ndarray:
    logger.info("Comienzo del embedding con UMAP")
    embedding_rf = UMAP(
    n_components=n_componentes,
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    learning_rate=learning_rate,
    metric=metric,
    random_state=SEMILLA,
    ).fit_transform(md)
    logger.info("Fin del embedding con UMAP")
    return embedding_rf


