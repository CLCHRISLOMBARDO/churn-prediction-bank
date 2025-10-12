#lgbm_optimizacion.py
import pandas as pd
import numpy as np
import lightgbm as lgb

from joblib import Parallel, delayed
import optuna
from optuna.study import Study
from time import time
import os

import pickle
import json
import logging
from optuna.samplers import TPESampler # Para eliminar el componente estocastico de optuna
from optuna.visualization import plot_param_importances, plot_contour,  plot_slice, plot_optimization_history

from src.config import *
ganancia_acierto = GANANCIA
costo_estimulo = ESTIMULO

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import json
import logging

# Reutilizo tus constantes/paths
# from src.config import SEMILLA, N_BOOSTS, N_FOLDS, db_path, best_iter_path, bestparams_path, GANANCIA, ESTIMULO
logger = logging.getLogger(__name__)

ganancia_acierto = GANANCIA
costo_estimulo   = ESTIMULO

# === Métrica custom de ganancia para XGBoost ===
def xgb_gan_eval(preds: np.ndarray, dtrain: xgb.DMatrix):
    """
    Reproduce tu lgb_gan_eval pero en interfaz XGBoost.
    preds: probabilidades (porque usamos binary:logistic)
    """
    weight = dtrain.get_weight()
    # misma lógica que en LGBM
    ganancia = np.where(weight == 1.00002, ganancia_acierto, 0) - np.where(weight < 1.00002, costo_estimulo, 0)
    ganancia = ganancia[np.argsort(preds)[::-1]]
    ganancia = np.cumsum(ganancia)
    # xgb espera (nombre, valor)
    return 'gan_eval', float(np.max(ganancia))

def optim_hiperp_binaria_xgb(
    X_train: pd.DataFrame,
    y_train_binaria: pd.Series,
    w_train: pd.Series,
    n_trials: int,
    fecha: str
):
    logger.info("Comienzo optimizacion hiperp binario (XGBoost)")
    name = "binaria_" + fecha

    # DMatrix con pesos
    dtrain = xgb.DMatrix(
        data=X_train,
        label=y_train_binaria.values,
        weight=w_train.values if w_train is not None else None
    )

    def objective(trial: optuna.trial.Trial) -> float:
        # Espacio de búsqueda (equivalentes comunes en XGB)
        max_depth         = trial.suggest_int('max_depth', 3, 12)
        eta               = trial.suggest_float('eta', 0.01, 0.2)                
        min_child_weight  = trial.suggest_float('min_child_weight', 1.0, 20.0)
        subsample         = trial.suggest_float('subsample', 0.5, 1.0)
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        gamma             = trial.suggest_float('gamma', 0.0, 10.0)
        reg_lambda        = trial.suggest_float('lambda', 0.0, 20.0)              # L2
        reg_alpha         = trial.suggest_float('alpha', 0.0, 10.0)               # L1

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',              # se ignora para selección de best; usamos feval custom
            'tree_method': 'hist',                 # rápido y estable
            'max_depth': max_depth,
            'eta': eta,
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'lambda': reg_lambda,
            'alpha': reg_alpha,
            'seed': SEMILLA,
            'verbosity': 0
        }

        # early stopping relacionado a lr (como hacías):
        es_rounds = int(50 + 5 / eta)

        cv_results = xgb.cv(
            params=params,
            dtrain=dtrain,
            num_boost_round=N_BOOSTS,
            nfold=N_FOLDS,
            stratified=True,
            shuffle=True,
            seed=SEMILLA,
            early_stopping_rounds=es_rounds,
            metrics=[],                    # sin métricas built-in
            custom_metric=xgb_gan_eval,    # <-- en vez de feval
            maximize=True                  # importante para tu ganancia
        )

        # Con feval='gan_eval', la columna se llama: 'test-gan_eval-mean'
        test_col = 'test-gan_eval-mean'
        if test_col not in cv_results.columns:
            # fallback defensivo
            raise RuntimeError(f"No se encontró la columna {test_col} en el CV de XGBoost")

        # mejor valor y mejor iter
        max_gan = cv_results[test_col].max()
        best_iter = int(cv_results[test_col].idxmax()) + 1  # 1-based

        # guardamos en user_attrs del trial
        trial.set_user_attr("best_iter", best_iter)

        # Devolvemos la métrica a maximizar (no hace falta multiplicar por N_FOLDS en XGB)
        return float(max_gan) * N_FOLDS

    storage_name = "sqlite:///" + db_path + "optimization_xgb.db"
    study_name = f"study_xgb_{name}"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        # sampler=optuna.samplers.TPESampler(seed=SEMILLA),  # opcional
    )

    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    best_iter   = study.best_trial.user_attrs["best_iter"]

    # Guardar best_iter
    try:
        with open(best_iter_path + f"best_iter_{name}.json", "w") as f:
            json.dump(best_iter, f, indent=4)
        logger.info(f"best_iter_{name}.json guardado en {best_iter_path}")
    except Exception as e:
        logger.error(f"Error al tratar de guardar el json de best iter: {e}")

    # Guardar best params
    try:
        with open(bestparams_path + f"best_params_{name}.json", "w") as f:
            json.dump(best_params, f, indent=4)
        logger.info(f"best_params_{name}.json guardado en {bestparams_path}")
        logger.info(f"Finalización de optimización XGB con study name {study_name}.")
    except Exception as e:
        logger.error(f"Error al tratar de guardar el json de los best parameters: {e}")

    return study


def graficos_bayesiana(study:Study, fecha: str):
    logger.info("Comienzo de la creacion de graficos")
    try:
        fig1 = plot_optimization_history(study)
        fig1.write_image(graf_bayesiana_path+f"{fecha}_graficos_opt_history.png")

        fig2 = plot_param_importances(study)
        fig2.write_image(graf_bayesiana_path+f"{fecha}_graficos_param_importances.png")

        fig3 = plot_slice(study)
        fig3.write_image(graf_bayesiana_path+f"{fecha}_graficos_slice.png")

        fig4 = plot_contour(study)
        fig4.write_image(graf_bayesiana_path+f"{fecha}_graficos_contour_all.png")

        fig5 = plot_contour(study, params=["num_leaves", "learning_rate"])
        fig5.write_image(graf_bayesiana_path+f"{fecha}_graficos_contour_specific.png")

        # logger.info(f" Gráficos guardados en {output_path}")
    except Exception as e:
        logger.error(f"Error al generar las gráficas: {e}")


def _ganancia_prob(y_hat:pd.Series|np.ndarray , y:pd.Series|np.ndarray ,prop=1,class_index:int =1,threshold:int=0.025)->float:
    logger.info("comienzo funcion ganancia con threhold = 0.025")
    @np.vectorize
    def _ganancia_row(predicted , actual , threshold=0.025):
        return (predicted>=threshold) * (ganancia_acierto if actual=="BAJA+2" else -costo_estimulo)
    logger.info("Finalizacion funcion ganancia con threhold = 0.025")
    return _ganancia_row(y_hat[:,class_index] ,y).sum() /prop


# def optim_hiperp_ternaria(X:pd.DataFrame|np.ndarray ,y:pd.Series|np.ndarray , n_trials:int , name:str)-> Study:
    
#     logger.info("Inicio de optimizacion hiperp ternario")
#     name ="ternaria"+name
#     def objective(trial):
#         max_depth = trial.suggest_int('max_depth', 2, 32)
#         min_samples_split = trial.suggest_int('min_samples_split', 2, 2000)
#         min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 200)
#         max_features = trial.suggest_float('max_features', 0.05, 0.7)

#         model = RandomForestClassifier(
#             n_estimators=N_ESTIMATORS,
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             min_samples_leaf=min_samples_leaf,
#             max_features=max_features,
#             max_samples=0.7,
#             random_state=SEMILLA,
#             n_jobs=12,
#             oob_score=True
#         )

#         model.fit(X, y)

#         return _ganancia_prob(model.oob_decision_function_, y)

#     storage_name = "sqlite:///" + db_path + "optimization_tree.db"
#     study_name = f"rf_ganancia_{name}"  

#     study = optuna.create_study(
#         direction="maximize",
#         study_name=study_name,
#         storage=storage_name,
#         load_if_exists=True,
#         sampler=TPESampler(seed=SEMILLA)
#     )

#     study.optimize(objective, n_trials=n_trials)

#     best_params = study.best_trial.params
    
#     try:
#         with open(bestparms_path+f"best_params_ganancia_{name}.json", "w") as f:
#             json.dump(best_params, f, indent=4) 
#             logger.info(f"best_params_ganancia_{name}.json guardado en outputs/optimizacion_rf/best_params/")
#         logger.info("Finalizacion de optimizacion hiperp binario.")
#     except Exception as e:
#         logger.error(f"Error al tratar de guardar el json de los best parameters por el error :{e}")
#     return study


