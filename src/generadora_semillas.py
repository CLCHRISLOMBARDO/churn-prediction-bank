import random
from src.config import SEMILLAS, N_SEMILLAS

semillas_nuevas=SEMILLAS.copy()
random.seed(42)

def create_semilla(n_semillas:int=N_SEMILLAS):
    while len(semillas_nuevas) < N_SEMILLAS -len(SEMILLAS):
        sem_nueva=random.randint(1,999999)
        if sem_nueva not in semillas_nuevas:
            semillas_nuevas.append(sem_nueva)
    return(semillas_nuevas)




