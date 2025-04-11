import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
from lib.ClassifierController import LlamaGuard3JailbreakJudge
import json
import os
from typing import List

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

def load_json_data(filename):
    """Charge les données JSON depuis un fichier."""
    with open(filename, 'r') as file:
        data = json.load(file)
        
    return data

# def extract_data(json_data):
#     """Extrait les données y_true, y_score, perturbation et nombre de copies."""
#     y_true = [req["jailbreak_groundtruth"] for req in json_data["requests"]]
#     y_score = [req["jailbreak_values"] for req in json_data["requests"]]
    
#     # Extraction des informations de la perturbation et du nombre de copies
#     name_info = json_data["name"]
#     perturbation = name_info.split("pert-")[1].split("_")[0]
#     nc = name_info.split("nc-")[1].split("_")[0]
    
#     return y_true, y_score, perturbation, nc

# def plot_pca_curve(embeddings, labels=None, perturbation, nc, output_path):

  
#     # Standardize the embeddings
#     scaler = StandardScaler()
#     embeddings_scaled = scaler.fit_transform(embeddings)
    
#     # Perform PCA to reduce to 2 dimensions
#     pca = PCA(n_components=2)
#     embeddings_pca = pca.fit_transform(embeddings_scaled)
    
#     # Plot the PCA results
#     plt.figure(figsize=(12, 8))
#     if labels is not None:
#         scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolor='k', s=50)
#         plt.colorbar(scatter, label="Labels")
#     else:
#         plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], color='blue', alpha=0.7, edgecolor='k', s=50)
    
#     plt.title(f"PCA of Embeddings - Perturbation: {perturbation}, Copies (nc): {nc}")
#     plt.xlabel("Principal Component 1")
#     plt.ylabel("Principal Component 2")
#     plt.grid(True)
    
#     # Save or show the plot
#     if output_path:
#         plt.savefig(output_path)
#         plt.close()
#     else:
#         plt.show()



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

        # Extract the first request
        if "requests" in json_data and json_data["requests"]:
            first_request = json_data["requests"][0]
        else:
            print(f"No requests found in {json_path}, skipping.")
            continue
        
        # Prepare output data
        output_data = {
            "name": json_data.get("name"),
            "model": json_data.get("model"),
            "data_set_path": json_data.get("data_set_path"),
            "data_set_type": json_data.get("data_set_type"),
            "classifier": json_data.get("classifier"),
            "timestamp": json_data.get("timestamp"),
            "first_request": first_request
        }    
#             # Extraire les informations nécessaires
#             y_true, y_score, perturbation, nc = extract_data(json_data)
            
#             # Définir le chemin de sauvegarde pour le graphique
        output_path = os.path.join(output_folder, f"{filename.split('.json')[0]}.json")
        
        
        with open(output_path, 'w') as out_f:
            json.dump(output_data, out_f, indent=4)
        print(f"Saved first request to {output_path}")

#             # Générer et sauvegarder la courbe ROC
#             plot_roc_curve(y_true, y_score, perturbation, nc, output_path)
#             print(f"Saved ROC curve for {filename} as {output_path}")




# Définir le dossier d'entrée et de sortie
input_folder = './output/pca'
output_folder = './graphs/pca'


# Générer et sauvegarder les courbes ROC


# Exemple d'utilisation
# generate_roc_curve_from_json("data.json")
generate_and_save_roc_curves(input_folder,output_folder)
# generate_roc_curve_from_json('/home/linux/Developpement/jailbreaktester/output/pert2detect_llama2_th-0.05_nc-5_pert-RandomSwapPerturbation_2024-11-11.json')

