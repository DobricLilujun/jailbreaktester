import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
from lib.ClassifierController import LlamaGuard3JailbreakJudge
import json
import os
from typing import List
from dotenv import load_dotenv

def load_json_data(filename):
    """Charge les données JSON depuis un fichier."""
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def extract_data(json_data):
    """Extrait les données y_true, y_score, perturbation et nombre de copies."""
    y_true = [req["jailbreak_groundtruth"] for req in json_data["requests"]]
    y_score = [req["jailbreak_values"] for req in json_data["requests"]]
    
    # Extraction des informations de la perturbation et du nombre de copies
    name_info = json_data["name"]
    perturbation = name_info.split("pert-")[1].split("_")[0]
    nc = name_info.split("nc-")[1].split("_")[0]
    
    return y_true, y_score, perturbation, nc

def plot_roc_curve(y_true, y_score, perturbation, nc, output_path):
    """Trace la courbe ROC avec les points correspondant aux différents seuils."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Création du graphique
    plt.figure(figsize=(12, 8))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    
    # Ajout des points pour chaque seuil
    for i, threshold in enumerate(thresholds):
        plt.plot(fpr[i], tpr[i], 'o', label=f"Threshold: {threshold:.2f}" if i % 2 == 0 else "")
    
    # Titre et légendes
    plt.title(f"ROC Curve - Perturbation: {perturbation}, Copies (nc): {nc}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    
    # Sauvegarder l'image dans le dossier de sortie
    plt.savefig(output_path)
    plt.close()

def generate_and_save_roc_curves(input_folder, output_folder):
    """Génère et sauvegarde les courbes ROC pour tous les fichiers JSON d'un dossier."""
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)
    
    # Parcourir tous les fichiers JSON du dossier d'entrée
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            json_path = os.path.join(input_folder, filename)
            
            # Charger les données JSON
            json_data = load_json_data(json_path)
            
            # Extraire les informations nécessaires
            y_true, y_score, perturbation, nc = extract_data(json_data)
            
            # Définir le chemin de sauvegarde pour le graphique
            output_path = os.path.join(output_folder, f"{filename.split('.json')[0]}.png")
            
            # Générer et sauvegarder la courbe ROC
            plot_roc_curve(y_true, y_score, perturbation, nc, output_path)
            print(f"Saved ROC curve for {filename} as {output_path}")


def evaluate_ground_truth_for_json_files(json_folder: str, api_key: str):
    """
    Évalue le ground truth pour tous les fichiers JSON dans un dossier donné à l'aide de LlamaGuard3JailbreakJudge
    et met à jour chaque fichier avec le résultat.
    
    Args:
    - json_folder (str): Le chemin vers le dossier contenant les fichiers JSON.
    - api_key (str): La clé API pour Hugging Face requise par LlamaGuard3JailbreakJudge.
    """
    # Initialise le classificateur avec la clé API
    judge = LlamaGuard3JailbreakJudge(api_key=api_key)

    # Parcourt tous les fichiers dans le dossier
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            print("Processing file: ", str(filename))
            file_path = os.path.join(json_folder, filename)
            
            # Charge les données du fichier JSON
            with open(file_path, 'r') as file:
                data = json.load(file)

            # Récupère les prompts et réponses
            prompts = [req["question"] for req in data["requests"]]
            responses = [req["response"] for req in data["requests"]]

            # Classe les réponses en utilisant LlamaGuard3JailbreakJudge
            classifications = judge(prompts, responses)

            # Met à jour le champ `jailbreak_groundtruth` pour chaque entrée
            for i, req in enumerate(data["requests"]):
                req["jailbreak_groundtruth"] = classifications[i]

            # Sauvegarde les données mises à jour dans le même fichier JSON
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)

            print(f"Fichier {filename} mis à jour avec les évaluations de ground truth.")

# Définir le dossier d'entrée et de sortie
input_folder = './output'
output_folder = './graphs'

# evaluate_ground_truth_for_json_files(input_folder, os.getenv("HUGGINGFACE_API_KEY"))

# Générer et sauvegarder les courbes ROC
generate_and_save_roc_curves(input_folder, output_folder)

# Exemple d'utilisation
# generate_roc_curve_from_json("data.json")

# generate_roc_curve_from_json('/home/linux/Developpement/jailbreaktester/output/pert2detect_llama2_th-0.05_nc-5_pert-RandomSwapPerturbation_2024-11-11.json')

