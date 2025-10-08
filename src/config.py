#config.py
## Configuracion General
SEMILLA= 773767#14
SEMILLAS=[259621, 282917, 413417, 773767, 290827]

## INPUT FILES
PATH_INPUT_DATA="data/competencia_01.csv"

## OUTPUTS FILES
PATH_OUTPUT_DATA="outputs/data_outputs/"
PATH_OUTPUT_OPTIMIZACION="outputs/lgbm_optimizacion/"
PATH_OUTPUT_LGBM="outputs/lgbm_model/"
PATH_NOTAS="notas/"

## Submuestra - solo uso por el momento el de segmentacion
MES_TRAIN =[202101,202102,202103]
MES_TEST =202104
MES_A_PREDECIR=202106

## OPTIMIZACION LGBM
UMBRAL=0.025
GANANCIA=780000
ESTIMULO = 20000
N_TRIALS= 150
N_BOOSTS=1000
N_FOLDS=5







