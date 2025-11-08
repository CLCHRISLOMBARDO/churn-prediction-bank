import numpy as np
import pandas as pd
import logging
import json

from src.config import *
from src.loader import cargar_datos
from src.preprocesamiento import conversion_binario,split_train_test_apred
from src.constr_lista_cols import contruccion_cols
from src.feature_engineering import feature_engineering_lag,feature_engineering_delta,feature_engineering_max_min,feature_engineering_ratio,feature_engineering_linreg,feature_engineering_normalizacion,feature_engineering_drop_cols,feature_engineering_rank
from src.lgbm_train_test import entrenamiento_lgbm,grafico_feature_importance,prediccion_test_lgbm ,calc_estadisticas_ganancia,grafico_curvas_ganancia, grafico_hist_ganancia ,preparacion_ypred_kaggle
## ---------------------------------------------------------Configuraciones Iniciales -------------------------------
## Carga de variables

logger=logging.getLogger(__name__)
