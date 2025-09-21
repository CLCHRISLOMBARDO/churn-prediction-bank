#optimizacion.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from joblib import Parallel, delayed
import optuna
from optuna.study import Study
from time import time
import datetime

import pickle
import json
import logging
from optuna.samplers import TPESampler # Para eliminar el componente estocastico de optuna

from src.config import PATH_OUTPUT_OPTIMIZACION, GANANCIA,ESTIMULO,SEMILLA ,N_ESTIMATORS

output_path = PATH_OUTPUT_OPTIMIZACION
db_path = output_path + 'db/'
bestparms_path = output_path+'best_params/'

ganancia_acierto = GANANCIA
costo_estimulo = ESTIMULO

logger = logging.getLogger(__name__)

def optim_hiperp_binaria(X:pd.DataFrame|np.ndarray ,y:pd.Series|np.ndarray , n_trials:int, name:str)-> Study:
    logger.info("Comienzo optimizacion hiperp binario")
    name ="binaria"+name
    

    def objective(trial):
        max_depth = trial.suggest_int('max_depth', 2, 32)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 2000)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 200)
        max_features = trial.suggest_float('max_features', 0.05, 0.7)

        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_samples=0.7,
            random_state=SEMILLA,
            n_jobs=12,
            oob_score=True
        )

        model.fit(X, y)

        proba_oob = model.oob_decision_function_
        pos_idx = int(np.where(model.classes_ == 1)[0][0])
        auc_score = roc_auc_score(y, proba_oob[:, pos_idx])


        return auc_score
    
    storage_name = "sqlite:///" + db_path + "optimization_tree.db"
    study_name = f"rf_auc_{name}"    

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        sampler=TPESampler(seed=SEMILLA)
    )

    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    
    # Mejor guardarlo en el main ?
    try:

        with open(bestparms_path+f"best_params_auc_{name}.json", "w") as f:
            json.dump(best_params, f, indent=4) 
        logger.info(f"best_params_auc_{name}.json guardado en outputs/optimizacion_rf/best_params/")
        logger.info("Finalizacion de optimizacion hiperp binario.")
    except Exception as e:
        logger.error(f"Error al tratar de guardar el json de los best parameters por el error :{e}")
    return study



def _ganancia_prob(y_hat:pd.Series|np.ndarray , y:pd.Series|np.ndarray ,prop=1,class_index:int =1,threshold:int=0.025)->float:
    logger.info("comienzo funcion ganancia con threhold = 0.025")
    @np.vectorize
    def _ganancia_row(predicted , actual , threshold=0.025):
        return (predicted>=threshold) * (ganancia_acierto if actual=="BAJA+2" else -costo_estimulo)
    logger.info("Finalizacion funcion ganancia con threhold = 0.025")
    return _ganancia_row(y_hat[:,class_index] ,y).sum() /prop


def optim_hiperp_ternaria(X:pd.DataFrame|np.ndarray ,y:pd.Series|np.ndarray , n_trials:int , name:str)-> Study:
    
    logger.info("Inicio de optimizacion hiperp ternario")
    name ="ternaria"+name
    def objective(trial):
        max_depth = trial.suggest_int('max_depth', 2, 32)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 2000)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 200)
        max_features = trial.suggest_float('max_features', 0.05, 0.7)

        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_samples=0.7,
            random_state=SEMILLA,
            n_jobs=12,
            oob_score=True
        )

        model.fit(X, y)

        return _ganancia_prob(model.oob_decision_function_, y)

    storage_name = "sqlite:///" + db_path + "optimization_tree.db"
    study_name = f"rf_ganancia_{name}"  

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        sampler=TPESampler(seed=SEMILLA)
    )

    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    
    try:
        with open(bestparms_path+f"best_params_ganancia_{name}.json", "w") as f:
            json.dump(best_params, f, indent=4) 
            logger.info(f"best_params_ganancia_{name}.json guardado en outputs/optimizacion_rf/best_params/")
        logger.info("Finalizacion de optimizacion hiperp binario.")
    except Exception as e:
        logger.error(f"Error al tratar de guardar el json de los best parameters por el error :{e}")
    return study


