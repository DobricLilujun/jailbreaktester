"""
File name: linker.py
Author: Nathan Foucher 
Contact: nathan.foucher@ext.uni.lu
Created: 30/09/2024
Version: 1.1
Description: Module to generate web application showing campaigns results.
"""

import datetime
import dateutil
import os
import json
from flask import Flask, render_template, request, jsonify, send_file
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import io
from sklearn.metrics import precision_score, recall_score, f1_score

app = Flask(__name__)


# Chargement des fichiers JSON depuis le dossier "output"
def load_json_files(directory="output"):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as f:
                data.append(json.load(f))
    return data


import numpy as np
import matplotlib.pyplot as plt


# Fonction pour générer des graphiques pour ASR, précision, rappel et F1-score avec Matplotlib
def graph_plot(campaigns):

    xlabel = "name"  # Can be changed to 'model', 'classifier'...

    metrics = []

    # Calculer les métriques pour chaque campagne
    for campaign in campaigns:
        metrics.append([campaign[xlabel], calculate_metrics(campaign)])

    # Initialisation des listes pour chaque métrique
    timestamps = []
    asr_classifier_values = []
    asr_groundtruth_values = []
    precision_values = []
    recall_values = []
    f1_values = []
    average_time_values = []

    # Boucle unique pour extraire toutes les valeurs en même temps
    for metric in metrics:
        timestamps.append(metric[0])
        asr_classifier_values.append(metric[1]["asr_classifier"])
        asr_groundtruth_values.append(metric[1]["asr_groundtruth"])
        precision_values.append(metric[1]["precision"])
        recall_values.append(metric[1]["recall"])
        f1_values.append(metric[1]["f1"])
        average_time_values.append(metric[1]["average_time"])

    # Palette de couleurs
    colors = plt.cm.viridis(np.linspace(0, 1, len(asr_classifier_values)))

    # Fonction utilitaire pour créer un graphique à barres avec des annotations
    def plot_metric(metric_values, metric_name, file_name):
        plt.figure(figsize=(10, 6))

        # Créer les barres avec des couleurs différentes
        bars = plt.bar(timestamps, metric_values, color=colors)

        # Ajouter les pourcentages ou valeurs au-dessus de chaque barre
        for bar, value in zip(bars, metric_values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{value:.2f}" if metric_name != "ASR" else f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.xlabel(xlabel)
        plt.ylabel(f"{metric_name} (%)" if metric_name == "ASR" else metric_name)
        plt.title(f"{metric_name} over Campaigns")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Sauvegarder le graphique
        plt.savefig(f"./static/{file_name}")
        plt.close()

    # Générer les graphiques pour ASR, précision, rappel et F1-score
    plot_metric(asr_classifier_values, "ASR", "asr_plot.png")
    plot_metric(asr_groundtruth_values, "ASR", "asr_groundtruth_plot.png")
    plot_metric(precision_values, "Precision", "precision_plot.png")
    plot_metric(recall_values, "Recall", "recall_plot.png")
    plot_metric(f1_values, "F1-Score", "f1_plot.png")
    plot_metric(average_time_values, "Average request time (s)", "average_time_plot.png")


# Route principale pour afficher les tests
@app.route("/")
def index():
    # Charger les données des fichiers JSON
    tests_data = load_json_files()

    # Récupérer les modèles et les classifiers uniques
    models = list(set(test["model"] for test in tests_data))
    classifiers = list(set(test["classifier"] for test in tests_data))
    dataset = list(set(test["data_set_path"] for test in tests_data))

    # Récupérer les paramètres de tri depuis la requête
    selected_model = request.args.get("model", "")
    selected_classifier = request.args.get("classifier", "")
    selected_dataset = request.args.get("dataset", "")

    # Filtrer les tests en fonction des critères sélectionnés
    if selected_model:
        tests_data = [test for test in tests_data if test["model"] == selected_model]
    if selected_classifier:
        tests_data = [
            test for test in tests_data if test["classifier"] == selected_classifier
        ]
    if selected_dataset:
        tests_data = [
            test for test in tests_data if test["data_set_path"] == selected_dataset
        ]

    tests_data = sorted(tests_data, key=lambda x: x["name"])

    return render_template(
        "index.html",
        tests=tests_data,
        models=models,
        classifiers=classifiers,
        datasets=dataset,
        selected_model=selected_model,
        selected_classifier=selected_classifier,
        selected_dataset=selected_dataset,
    )


def calculate_metrics(campaign):
    total_request = 0
    successful_jailbreaks_classifier = 0

    for request in campaign["requests"]:
        total_request += 1
        if request.get("jailbreak_successful", False):
            successful_jailbreaks_classifier += 1

    asr_classifier = (successful_jailbreaks_classifier / total_request) * 100 if total_request > 0 else 0

    # try:
    true_labels = []
    predicted_labels = []
    timestamps = []
    successful_jailbreaks_ground_truth = 0 
    VP = FP = FN = VN = 0
    # Extraire les vraies étiquettes (si jailbreak réussi ou non) et les étiquettes prédites (étiquettes actuelles)
    for request in campaign["requests"]:
        # true_labels.append(
        #     1 if request["jailbreak_groundtruth"] else 0
        # )  # Remplace par la vraie valeur attendue
        # predicted_labels.append(1 if request["jailbreak_successful"] else 0)

        try :
            predicted_success = request.get(
                "jailbreak_successful", False
            )  # Prédiction du modèle
            actual_success = request.get(
                "jailbreak_groundtruth", False
            )  # Réalité (il faudrait ajouter ces données si elles ne sont pas présentes)


            if actual_success :
                successful_jailbreaks_ground_truth += 1

            if predicted_success and actual_success:
                VP += 1  # Vrai Positif
            elif predicted_success and not actual_success:
                FP += 1  # Faux Positif
            elif not predicted_success and actual_success:
                FN += 1  # Faux Négatif
            else:
                VN += 1  # Vrai Négatif
        except :
            VP, FP, FN, VN, successful_jailbreaks_ground_truth   = 0, 0, 0, 0, 0

        timestamps.append(dateutil.parser.parse(request.get("timestamp")))
        # Calcul des différences entre chaque timestamp consécutif
        time_deltas = [
            (timestamps[i+1] - timestamps[i]).total_seconds()
            for i in range(len(timestamps) - 1)
        ]
        asr_groudTruth = (successful_jailbreaks_ground_truth / total_request) * 100 if total_request > 0 else 0

        # Calcul de la moyenne des temps de requête
    average_time = sum(time_deltas) / len(time_deltas) if time_deltas else 0
    # except:
    #     VP, FP, FN, VN,average_time  = 0, 0, 0, 0, 0

    precision = VP / (VP + FP) if (VP + FP) > 0 else 0
    recall = VP / (VP + FN) if (VP + FN) > 0 else 0
    f1 = (2 * VP) / (2 * VP + FP + FN) if (VP + FP + FN) > 0 else 0

    return {
        "total_tests": total_request,
        "successful_jailbreaks_classifier": successful_jailbreaks_classifier,
        "asr_classifier": round(asr_classifier, 2),
        "successful_jailbreaks_groundtruth": successful_jailbreaks_ground_truth,
        "asr_groundtruth": asr_groudTruth,
        "VP": VP,
        "FP": FP,
        "FN": FN,
        "VN": VN,
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2),
        "average_time": round(average_time, 2),
    }


@app.route("/export_data")
def export_data():
    # Récupérer les timestamps sélectionnés
    timestamps = request.args.getlist("timestamps")

    # Charger les données des fichiers JSON
    tests_data = load_json_files()

    # Filtrer les tests en fonction des timestamps sélectionnés
    selected_campaigns = [
        test for test in tests_data if test["timestamp"] in str(timestamps)
    ]

    metrics = []

    for campaign in selected_campaigns:
        metrics.append([campaign["name"], calculate_metrics(campaign)])

    return jsonify(metrics)


@app.route("/test/<timestamp>")
def test_details(timestamp):
    tests_data = load_json_files()

    # Trouver le test avec le bon timestamp
    test_data = next(
        (test for test in tests_data if test["timestamp"] == timestamp), None
    )
    if test_data is None:
        return "Test not found", 404

    # Calcul des métriques ASR + Precision, Recall, F1
    metrics = calculate_metrics(test_data)

    # Préparation des données pour le graphique (ASR pour chaque test)
    labels = [f"Test {i+1}" for i in range(len(test_data["requests"]))]

    return render_template(
        "test_details.html", test=test_data, metrics=metrics, labels=labels
    )


@app.route("/update_graph")
def update_graph():
    # Récupérer les timestamps sélectionnés
    timestamps = request.args.getlist("timestamps")
    # print("Selected Timestamps:", timestamps)

    # Charger les données des fichiers JSON
    tests_data = load_json_files()

    # Filtrer les tests en fonction des timestamps sélectionnés
    selected_campaigns = [
        test for test in tests_data if test["timestamp"] in str(timestamps)
    ]

    # Vérifiez que des tests ont été sélectionnés
    if not selected_campaigns:
        print("Aucun test sélectionné pour le graphique.")  # Debug
        return
    
    selected_campaigns = sorted(selected_campaigns, key=lambda x: x["name"])

    # Générer le graphique ASR avec les campagnes sélectionnés
    graph_plot(selected_campaigns)

    # Calculer les statistiques d'accord/désaccord entre les classifiers
    agreement_stats = calculate_agreement(selected_campaigns)

    return jsonify(success=True, agreement_stats=agreement_stats)


def calculate_agreement(selected_campaigns):
    classifier_results = {}

    # Collecter les résultats par classifier
    for campaign in selected_campaigns:
        classifier_results[campaign["timestamp"]] = [
            req["jailbreak_successful"] for req in campaign["requests"]
        ]

    # S'assurer qu'il y a au moins deux campagnes à comparer
    if len(classifier_results) < 2:
        return {"agreement_percentage": "", "disagreement_percentage": ""}

    # Calculer l'accord et le désaccord entre les campagnes
    total_comparisons = 0
    total_agreements = 0

    # Comparer les résultats entre classifiers
    classifiers = list(classifier_results.keys())
    try:
        for i in range(len(classifiers) - 1):
            for j in range(i + 1, len(classifiers)):
                classifier_i_results = classifier_results[classifiers[i]]
                classifier_j_results = classifier_results[classifiers[j]]

                # Si pas le meme nombre de tests
                if len(classifier_i_results) != len(classifier_j_results):
                    return {"agreement_percentage": "", "disagreement_percentage": ""}

                for k in range(len(classifier_i_results)):
                    if classifier_i_results[k] == classifier_j_results[k]:
                        total_agreements += 1
                    total_comparisons += 1
    except:
        print("error while computing agrement")
        return {"agreement_percentage": "?", "disagreement_percentage": "?"}

    # Calculer les pourcentages
    agreement_percentage = (
        (total_agreements / total_comparisons) * 100 if total_comparisons > 0 else 100
    )
    disagreement_percentage = 100 - agreement_percentage

    return {
        "agreement_percentage": round(agreement_percentage, 2),
        "disagreement_percentage": round(disagreement_percentage, 2),
    }


def run_flask():
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    app.run(debug=True)
