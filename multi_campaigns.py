import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

dict_pert = ["RandomSwapPerturbation", "RandomPatchPerturbation", "RandomInsertPerturbation", "GPTParaphrasePerturbation"]

# Paramètres à varier
# thresholds = [round(i * 0.05, 2) for i in range(1, 20)]  # de 0.05 à 0.95
thresholds = [0.05]
num_copies = range(5, 11)  # de 5 à 10

# Paramètres fixes
llm_model = '[LOCAL] Ollama-llama2:7b-chat'
data_set_type = 'JailBreakBench Data Set'
data_set_path = '/home/linux/Developpement/jailbreaktester/dataset/jailbreakBench_full_success.json'
classifier = 'Pert2Detect'
classifier_options_template = '--threshold {threshold} --smoothllm_num_copies {num_copies} --smoothllm_pert_types ["{pert_type}"] --smoothllm_pert_pct_min 5 --smoothllm_pert_pct_max 15'

# Nombre maximal de tentatives
MAX_RETRIES = 3
# Temps d'attente avant réessai en secondes (1 minute)
RETRY_DELAY = 60

def run_campaign(name, llm_model, data_set_type, data_set_path, classifier, classifier_options):
    command = [
        'python', 'main.py',
        '--name', name,
        '--llm_model', llm_model,
        '--data_set_type', data_set_type,
        '--data_set_path', data_set_path,
        '--classifier', classifier,
        '--classifier_options', classifier_options
    ]

    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            # Exécution de la commande
            subprocess.run(command, check=True)  # check=True pour lever une erreur si le processus échoue
            print(f"Campagne '{name}' terminée avec succès.")
            return  # Si succès, on sort de la fonction
        except subprocess.CalledProcessError as e:
            attempt += 1
            print(f"Échec de la campagne '{name}' (tentative {attempt}/{MAX_RETRIES}). Erreur: {e}")
            if attempt < MAX_RETRIES:
                print(f"Nouvelle tentative pour la campagne '{name}' dans {RETRY_DELAY / 60} minutes.")
                time.sleep(RETRY_DELAY)
            else:
                print(f"Campagne '{name}' a échoué après {MAX_RETRIES} tentatives.")

# Exécute les campagnes de tests avec un maximum de 10 threads
with ThreadPoolExecutor(max_workers=1) as executor:
    futures = []
    for threshold in thresholds:
        for pert_type in dict_pert :
            for copies in num_copies:
                # Créer le nom du fichier
                name = f'pert2detect_llama2_th-{threshold}_nc-{copies}_pert-{pert_type}'
                
                # Préparer les options du classificateur
                classifier_options = classifier_options_template.format(threshold=threshold, num_copies=copies, pert_type=pert_type)
                
                # Soumettre la tâche au thread pool
                futures.append(executor.submit(run_campaign, name, llm_model, data_set_type, data_set_path, classifier, classifier_options))
    
    # Attendre que toutes les campagnes soient terminées
    for future in as_completed(futures):
        try:
            future.result()  # Obtenir le résultat (ou une exception si elle a été levée)
        except Exception as e:
            print(f"Erreur non gérée dans une campagne : {e}")
