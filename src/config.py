#config.py
## Configuracion General
SEMILLA= 773767#14
SEMILLAS=[259621, 282917, 413417, 773767, 290827]
N_SEMILLAS= 49
## INPUT PATHS
PATH_INPUT_DATA="data/competencia_01.csv"

## LOG PATH
PATH_LOGS= "logs/"

## OUTPUTS PATHS
PATH_OUTPUT_DATA="outputs/data_outputs/"
PATH_OUTPUT_FINALES = "outputs/finales_outputs/"
PATH_OUTPUT_EXPERIMENTOS = "outputs/experimentos_outputs/"

    # PATH_OUTPUT_FINALES
## BAYESIANA PATHS
PATH_OUTPUT_OPTIMIZACION=PATH_OUTPUT_FINALES+"lgbm_optimizacion/"
db_path = PATH_OUTPUT_OPTIMIZACION + 'db/'
bestparams_path = PATH_OUTPUT_OPTIMIZACION+'best_params/'
best_iter_path = PATH_OUTPUT_OPTIMIZACION+'best_iters/'
graf_bayesiana_path = PATH_OUTPUT_OPTIMIZACION+'grafico_bayesiana/'

## MODEL PATHS
PATH_OUTPUT_LGBM=PATH_OUTPUT_FINALES+"lgbm_model/"
model_path=PATH_OUTPUT_LGBM + 'model/'
prediccion_final_path = PATH_OUTPUT_LGBM + 'final_prediction/'
graf_curva_ganancia_path=PATH_OUTPUT_LGBM +'graf_curva_ganancia/'
graf_hist_ganancia_grilla_path=PATH_OUTPUT_LGBM +'graf_ganancias_hist_grilla/'
graf_hist_ganancia_total_path=PATH_OUTPUT_LGBM +'graf_ganancias_hist_total/'
graf_hist_ganancia_semillas_path=PATH_OUTPUT_LGBM +'graf_ganancias_hist_semillas/'
umbrales_path=PATH_OUTPUT_LGBM +'umbrales/'
feat_imp_path=PATH_OUTPUT_LGBM +'feature_importances/'

    # PATH_OUTPUT_EXPERIMENTOS
## EXPERIMENTS PATHS
path_output_exp_model=PATH_OUTPUT_EXPERIMENTOS + "exp_model/"
path_output_exp_feat_imp=PATH_OUTPUT_EXPERIMENTOS + "exp_feat_importances/"
path_output_exp_graf_gan_hist_grilla=PATH_OUTPUT_EXPERIMENTOS + "exp_graf_ganancias_hist_grilla/"
path_output_exp_graf_gan_hist_total=PATH_OUTPUT_EXPERIMENTOS + "exp_graf_ganancias_hist_total/"
path_output_exp_graf_gan_hist_semillas=PATH_OUTPUT_EXPERIMENTOS + "exp_graf_ganancias_hist_semillas/"
path_output_exp_graf_curva_ganancia=PATH_OUTPUT_EXPERIMENTOS + "exp_graf_curva_ganancia/"
path_output_exp_umbral=PATH_OUTPUT_EXPERIMENTOS + "exp_umbrales/"


## Submuestra - solo uso por el momento el de segmentacion
MES_TRAIN =[202101,202102,202103]
MES_TEST =202104
MES_A_PREDECIR=202106
MES_01=202101
MES_02=202102
MES_03=202103
MES_04=202104
MES_05=202105

## OPTIMIZACION LGBM
UMBRAL=0.025
GANANCIA=780000
ESTIMULO = 20000
N_TRIALS= 30
N_BOOSTS=1000
N_FOLDS=5





