"""
File name: synchronize_ground_truth.py
Author: Nathan Foucher 
Contact: nathan.foucher@ext.uni.lu
Created: 30/09/2024
Version: 1.1
Description: Python module used to generate ground truth file based on several campaigns (on the same dataset!) containing groundtruth in a choosen folder. 
"""
import os
import json

def generate_ground_truth(json_folder: str, oracle_fct: callable, output :str = 'output.json'):
    """
    Recupere les valeurs de groundtruth dans les json de campagnes presents dans le dossier source. Utilise la fonction d'evaluation 
    pour prendre une decision sur le groundtruth final.

    Genere en sortie un fichier json de campagnes contenant le ground truth final evalue. 

    Args:
    - json_folder (str): Le chemin vers le dossier contenant les fichiers JSON.
    - oracle_fct (function): Fonction d'evaluation du ground truth.
    """

    # Parcourt tous les fichiers dans le dossier
    all_gt = []
    questions = []
    data_set = ""
    data_set_type = ""
    
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            print("Processing file: ", str(filename))
            file_path = os.path.join(json_folder, filename)

            # Charge les données du fichier JSON
            with open(file_path, "r") as file:
                data = json.load(file)
            
            # Save the questions
            if len(questions) == 0 :
                questions = [req["question"] for req in data["requests"]]
                data_set_path = data["data_set_path"]
                data_set_type = data["data_set_type"]
            # # Verify the files
            # else :
            #     if data_set_path != data["data_set_path"] or data_set_type != data["data_set_type"]:
            #         raise ValueError("The datasets in both files must be the same.")
            #     for req, question in zip(data["requests"], questions):
            #         if req["question"] != question:
            #             raise ValueError(f"Questions do not match: '{str(req['question'])}' vs '{question}'")
            
            # Récupère les valeurs du groundtruth
            gt = [req["jailbreak_groundtruth"] for req in data["requests"]]
            
            all_gt.append(gt)

    

    # Transpose list 
    all_gt= list(map(list, zip(*all_gt)))
    # Process list
    all_gt = oracle_fct(all_gt)

    # Construction du dictionnaire JSON
    json_data = {
        "name": f"[GT_2] {data_set_path.split('/')[-1]}" ,
        "data_set_path": data_set_path,
        "data_set_type": data_set_type,
        "requests": [
            {"question": question, "jailbreak_groundtruth_": groundtruth}
            for question, groundtruth in zip(questions, all_gt)
        ]
    }

    # Save the file
    with open(output, "w") as file:
        json.dump(json_data, file, indent=4)

    print(f"File {output} successfully saved !")


def synchronize_all_files_in_folder(folder_path, json_with_ground_truth):
    """
    Applique la fonction synchronize_ground_truth sur tous les fichiers JSON dans un dossier
    sauf le fichier `json_with_ground_truth`.
    """
    # Parcourir tous les fichiers du dossier
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Vérifier si le fichier est un fichier JSON et n'est pas le fichier de référence
        if filename.endswith('.json') and file_path != json_with_ground_truth:
            print(f"Synchronizing ground truth in: {filename}")
            synchronize_ground_truth(json_with_ground_truth, file_path)


def synchronize_ground_truth(json_with_ground_truth: str, json_without_ground_truth: str):
    """
    Synchronize the groundtruth values from one JSON file to another.

    Args:
        json_with_ground_truth (str): Path to the JSON file containing groundtruth values.
        json_without_ground_truth (str): Path to the JSON file without groundtruth values.

    Raises:
        ValueError: If the files are not compatible (different dataset paths or types, or missing groundtruth).
    """
    
    # Load both JSON files
    with open(json_with_ground_truth, 'r') as f:
        data_with_gt = json.load(f)
    
    with open(json_without_ground_truth, 'r') as f:
        data_without_gt = json.load(f)
    
    # Check if the datasets match
    if data_with_gt["data_set_path"] != data_without_gt["data_set_path"] or \
       data_with_gt["data_set_type"] != data_without_gt["data_set_type"]:
        raise ValueError("The datasets in both JSON files must be the same.")
    
    # Check that the first file has the groundtruth information
    for request in data_with_gt["requests"]:
        if "jailbreak_groundtruth" not in request:
            raise ValueError(f"Missing 'jailbreak_groundtruth' in the file: {json_with_ground_truth}")
    
    # Synchronize the groundtruth values
    for req_with_gt, req_without_gt in zip(data_with_gt["requests"], data_without_gt["requests"]):
        if req_with_gt["question"] != req_without_gt["question"]:
            raise ValueError(f"Questions do not match: '{req_with_gt['question']}' vs '{req_without_gt['question']}'")
        
        # Add groundtruth to the file missing it
        req_without_gt["jailbreak_groundtruth"] = req_with_gt["jailbreak_groundtruth"]
    
    # Save the synchronized results back to the second file
    with open(json_without_ground_truth, 'w') as f:
        json.dump(data_without_gt, f, indent=4)
    
    print(f"Synchronized groundtruth values from '{json_with_ground_truth}' to '{json_without_ground_truth}'.")

def majority_bool_per_column(matrix):
    # Calculate the majority bool for each column
    return [sum(row) > len(row) / 2 for row in matrix]

def only_totaly_false(matrix):
    # say false for prompts with full false evaluation
    print(matrix)
    return [sum(row) > 1 for row in matrix]




def print_results_successful_non_successful(file_path):
    # Load JSON data from file
    with open(file_path, 'r') as file:
        data = json.load(file)

    successful_jailbreaks = []
    non_successful_jailbreaks = []

    # Extract requests and filter where "jailbreak_successful" is true

    for request in data.get("requests", []):
        if isinstance(request, dict):
            if request.get("jailbreak_groundtruth_") is True:
                successful_jailbreaks.append(request)
            else:
                non_successful_jailbreaks.append(request)


    # Convert the results to JSON format for readability
    result = {
        "successful_jailbreaks_count": len(successful_jailbreaks),
        "non_successful_jailbreaks_count": len(non_successful_jailbreaks),

        "successful_jailbreaks": successful_jailbreaks,
        "non_successful_jailbreaks": non_successful_jailbreaks
    }

    # Convert the filtered requests to JSON format
    filtered_requests = json.dumps(result, indent=4)

    return result

    # # Example usage
    # file_path = '/content/sample_data/[GT_2] dataset_JailBreakBench_failed_and_successfull.json'  # replace with your JSON file path
    #requests = filter_successful_jailbreaks(file_path)

    # # Print or process the filtered requests
    #  print(requests)


    






output_file = 'output_GT.json'

#generate_ground_truth('./output/vllm_groundtruth_5', only_totaly_false, output_file)
#synchronize_all_files_in_folder('./output/vllm_groundtruth_5', output_file)

requests = print_results_successful_non_successful('./ground_truth/[GT_2] dataset_JailBreakBench_failed_and_successfull.json')
print("Successful jailbreaks count:", requests["successful_jailbreaks_count"])
print("Non-successful jailbreaks count:", requests["non_successful_jailbreaks_count"])
print("Total number of prompts: ", requests["successful_jailbreaks_count"]+ requests["non_successful_jailbreaks_count"])