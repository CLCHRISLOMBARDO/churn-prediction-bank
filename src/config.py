# src/config.py
import os
import yaml
import logging

logger = logging.getLogger(__name__)

# Ruta del archivo de configuración (ajustá si lo tenés en otro lado)
PATH_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

try:
    with open(PATH_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
        exp        = cfg["configuracion_experimentos"]
        bayes      = cfg["configuracion_bayesiana"]
        sem        = cfg["configuracion_semilla"]
        gcp        = cfg["configuracion_gcp"]
        paths      = cfg["configuracion_paths"]
        inp        = paths["path_inputs"]
        out        = paths["path_outputs"]
        out_bayes  = paths["path_outputs_bayesian"]
        out_final  = paths["path_outputs_finales"]
        out_exp    = paths["path_outputs_experimentos"]
        


        COMPETENCIA =cfg["COMPETENCIA"]
        PROCESO_PPAL = cfg["PROCESO_PPAL"]


        if COMPETENCIA == 1:
            comp    = cfg["configuracion_competencia_1"]
        elif COMPETENCIA == 2:
            comp      = cfg["configuracion_competencia_2"]

        # =================== Optimización LGBM ===================
        N_BAYESIANA         = bayes.get("N_BAYESIANA")
        UMBRAL              = bayes.get("UMBRAL", 0.025)
        GANANCIA            = bayes.get("GANANCIA", 780000)
        ESTIMULO            = bayes.get("ESTIMULO", 20000)
        N_TRIALS            = bayes.get("N_TRIALS", 35)
        N_BOOSTS            = bayes.get("N_BOOSTS", 1000)
        N_FOLDS             = bayes.get("N_FOLDS", 5)
        MES_VAL_BAYESIANA   = bayes.get("MES_VAL_BAYESIANA", 202103) 

        if PROCESO_PPAL =="bayesiana":
            N_EXPERIMENTO = bayes["N_BAYESIANA"]
        else:
            N_EXPERIMENTO = exp["N_EXP"]
            TOP_MODELS = exp["TOP_MODELS"]
        

        # ================= Configuración General =================
        SEMILLA    = sem.get("SEMILLA", 773767)
        SEMILLAS   = sem.get("SEMILLAS", [259621, 282917, 413417, 773767, 290827])
        N_SEMILLAS = sem.get("N_SEMILLAS", 49)
        

        # ---------------- Entorno (GCP vs local) ----------------
        in_gcp = bool(gcp.get("IN_GCP", False))
        if in_gcp:
            PLACE_PATHS = paths["place_path"]["GCP_PATH"]
        else:
            PLACE_PATHS = paths["place_path"]["LOCAL_PATH"]

        # ================= Rutas de INPUT / LOG ==================
        PATH_INPUT_DATA = PLACE_PATHS + inp["PATH_INPUT_DATA"]
        FILE_INPUT_DATA = PLACE_PATHS + comp["FILE_INPUT_DATA"]
        FILE_INPUT_DATA_CRUDO = PLACE_PATHS + comp["FILE_INPUT_DATA_CRUDO"]
        PATH_LOGS       = PLACE_PATHS + paths["PATH_LOGS"]
        PATH_LOG_GLOBAL = paths["PATH_LOG_GLOBAL"]

        # ==================== OUTPUTS BASES ======================
        PATH_OUTPUT              = PLACE_PATHS + out["PATH_OUTPUT"]
        PATH_OUTPUT_DATA         = PLACE_PATHS + out["PATH_OUTPUT_DATA"]
        PATH_OUTPUT_BAYESIAN     = PLACE_PATHS + out["PATH_OUTPUT_BAYESIAN"]
        PATH_OUTPUT_FINALES      = PLACE_PATHS + out["PATH_OUTPUT_FINALES"]
        PATH_OUTPUT_EXPERIMENTOS = PLACE_PATHS + out["PATH_OUTPUT_EXPERIMENTOS"]
        PATH_OUTPUT_EDA = PLACE_PATHS + out["PATH_OUTPUT_EDA"]

        # ============= PATH_OUTPUT_BAYESIAN (detallados) =========
        path_output_bayesian_db         = PLACE_PATHS + out_bayes["PATH_OUTPUT_BAYESIAN_DB"]
        path_output_bayesian_bestparams = PLACE_PATHS + out_bayes["PATH_OUTPUT_BAYESIAN_BESTPARAMS"]
        path_output_bayesian_best_iter  = PLACE_PATHS + out_bayes["PATH_OUTPUT_BAYESIAN_BEST_ITER"]
        path_output_bayesian_graf       = PLACE_PATHS + out_bayes["PATH_OUTPUT_BAYESIAN_GRAF"]

        # =============== PATH_OUTPUT_FINALES (detallados) ========
        path_output_finales_model        = PLACE_PATHS + out_final["PATH_OUTPUT_FINALES_MODEL"]
        path_output_finales_feat_imp     = PLACE_PATHS + out_final["PATH_OUTPUT_FINALES_FEAT_IMP"]
        path_output_prediccion_final     = PLACE_PATHS + out_final["PATH_OUTPUT_FINALES_PREDICCION_FINAL"]

        # ======= PATH_OUTPUT_EXPERIMENTOS (derivados directos) ===
        path_output_exp_model                  = PLACE_PATHS + out_exp["PATH_OUTPUT_EXP_MODEL"]
        path_output_exp_feat_imp               = PLACE_PATHS + out_exp["PATH_OUTPUT_EXP_FEAT_IMP"]
        path_output_exp_graf_gan_hist_grilla   = PLACE_PATHS + out_exp["PATH_OUTPUT_EXP_GRAF_GAN_HIST_GRILLA"]
        path_output_exp_graf_gan_hist_total    = PLACE_PATHS + out_exp["PATH_OUTPUT_EXP_GRAF_HIST_TOTAL"]
        path_output_exp_graf_gan_hist_semillas = PLACE_PATHS + out_exp["PATH_OUTPUT_EXP_GRAF_HIST_SEMILLA"]
        path_output_exp_graf_curva_ganancia    = PLACE_PATHS + out_exp["PATH_OUTPUT_EXP__GRAF_CURVA_GANANCIA"]
        path_output_exp_umbral                 = PLACE_PATHS + out_exp["PATH_OUTPUT_EXP_UMBRAL"]

        # ================= Submuestra / competencia ==============
        MES_TRAIN      = comp.get("MES_TRAIN", [202101, 202102, 202103])
        MES_TEST       = comp.get("MES_TEST", [202104])
        MES_A_PREDECIR = comp.get("MES_A_PREDECIR", 202106)
        MES_01 = comp.get("MES_01", 202101)
        MES_02 = comp.get("MES_02", 202102)
        MES_03 = comp.get("MES_03", 202103)
        MES_04 = comp.get("MES_04", 202104)
        MES_05 = comp.get("MES_05", 202105)



except Exception as e:
    logger.error(f"Error al cargar el archivo de configuración: {e}")
    raise
