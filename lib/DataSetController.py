"""
File name: DataSetController.py
Author: Nathan Foucher 
Contact: nathan.foucher@ext.uni.lu
Created: 30/09/2024
Version: 1.0
Description: File where is defined the abstract class to communicate with the datasets. You need to define your own implementations of the abstracts methods in this file. 
"""

from abc import ABC, abstractmethod
import json
from typing import Tuple
import csv
import json


class DataSetController(ABC):

    @abstractmethod
    def __init__(self, path_to_data_set: str):
        """
        Initializes the class with a file of questions.

        Args:
            path_to_data_set (str): The path to the dataset file.
        """
        self.current_index = 0
        pass

    @abstractmethod
    def nextPrompt(self) -> str:
        """
        Returns the next question (prompt) from the file.

        Returns:
            str: A string representing the question.
        """
        pass

    @abstractmethod
    def getCurrentIndex(self) -> int:
        """
        Sends the current index.

        Returns:
            int: Current index, -1 if the end of the file is reached.
        """
        return self.current_index

    @abstractmethod
    def getNumberOfPrompts(self) -> int:
        """
        Sends the total number of prompts.

        Returns:
            int: Total number of prompts.
        """
        pass


class GroundTruth(DataSetController):
    """
    Concrete class that interacts with a dataset of questions stored in a JSON file.
    """

    def __init__(self, path_to_data_set: str):
        """
        Initializes the JSONDataSet with the path to the JSON file and loads its content.

        Args:
            path_to_data_set (str): Path to the dataset file (JSON).
        """
        self.current_index = 0
        self.path_to_data_set = path_to_data_set
        self.data = {}

        # Load the JSON dataset
        self.load_data(path_to_data_set)

    def load_data(self, path: str):
        """
        Loads the dataset from a JSON file.

        Args:
            path (str): Path to the JSON file.
        """
        try:
            with open(path, "r") as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            self.data = {}

    def nextPrompt(self) -> str:
        """
        Returns the next question (prompt) from the dataset along with its jailbreak classification.

        Returns:
            str: Text of the auestion
        """
        requests = self.data.get("requests", [])
        if self.current_index < len(requests):
            prompt_data = requests[self.current_index]

            return prompt_data.get("question", "")
        else:
            self.current_index = -1
            return ""

    def getCurrentIndex(self) -> int:
        """
        Sends the current index in the dataset.

        Returns:
            int: Current index, -1 if the end of the file is reached.
        """
        return super().getCurrentIndex()

    def getNumberOfPrompts(self) -> int:
        """
        Sends the total number of prompts in the dataset.

        Returns:
            int: Total number of prompts.
        """
        return len(self.data.get("requests", []))

    def record_ground_truth(self, question: str, groundtruth: list[bool]):
        """
        Updates the 'jailbreak_groundtruth' value for a specific question in the dataset and saves the change.

        Args:
            question (str): The question whose groundtruth value should be updated.
            groundtruth (list[bool]): The new groundtruth value (True/False).
        """
        requests = self.data.get("requests", [])

        requests[self.current_index]["jailbreak_groundtruth"] = groundtruth[0]
        
        self.current_index += 1

        # Write the updated data back to the original file
        try:
            with open(self.path_to_data_set, "w") as f:
                json.dump(self.data, f, indent=4)
            print(f"Groundtruth for the question updated successfully.")
        except Exception as e:
            print(f"Error writing to JSON file: {e}")


class SimpleJsonDataSet(DataSetController):
    def __init__(self, path_to_data_set):
        """
        Initialise la classe avec un fichier JSON contenant des prompts.

        Args:
            path_to_data_set (str): Le chemin vers le fichier JSON.

        Returns:
            bool: Retourne True si l'initialisation est réussie, False sinon.
        """
        self.data = {}
        self.current_index = 0
        try:
            with open(path_to_data_set, "r") as file:
                json_data = json.load(file)
                # S'assurer que la clé 'prompts' existe
                if "prompts" in json_data:
                    self.prompt_number = len(json_data["prompts"])
                    self.data = json_data["prompts"]
                    self.current_index = 1
                else:
                    raise Exception(f"Error: malformed JSON '{path_to_data_set}'")
        except FileNotFoundError as e:
            raise Exception(f"Error: unable to find the file '{path_to_data_set}'")
        except json.JSONDecodeError:
            raise Exception(f"Error: malformed JSON '{path_to_data_set}'")

    def nextPrompt(self) -> str:
        """
        Renvoie la prochaine question (prompt) du fichier JSON, ainsi que son index.

        Returns:
            str: contient la question
        """
        if str(self.current_index) in self.data:
            prompt = self.data[str(self.current_index)]
            self.current_index += 1
            return str(prompt)
        else:
            self.current_index = -1
            return None

    def getCurrentIndex(self) -> int:
        return super().getCurrentIndex()

    def getNumberOfPrompts(self) -> int:
        return self.prompt_number


class SimpleCSVDataSet(DataSetController):

    def __init__(self, path_to_data_set: str):
        """
        Initialise la classe avec un fichier CSV et charge les questions.

        Args:
            path_to_data_set (str): Le chemin vers le fichier de données (dataset).
        """
        self.path_to_data_set = path_to_data_set
        self.prompts = []
        self.current_index = 0
        self._load_csv()

    def _load_csv(self):
        """
        Charge les données à partir du fichier CSV et stocke les prompts.
        """
        try:
            with open(self.path_to_data_set, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Combine jailbreak_query et redteam_query en un seul prompt
                    self.prompts.append(row["jailbreak_query"])
        except Exception as e:
            raise Exception(f"Error : Unable to load csv '{self.path_to_data_set}'")

    def nextPrompt(self) -> str:
        """
        Renvoie la prochaine question (prompt) et son index.

        Returns:
            str: La prochaine question du dataset.
        """
        if self.current_index < len(self.prompts):
            prompt = self.prompts[self.current_index]
            self.current_index += 1
            return str(prompt)
        else:
            self.current_index = -1
            return None

    def getCurrentIndex(self) -> int:
        """
        Renvoie l'index courant.

        Returns:
            int: Index courant, -1 si fin du fichier.
        """
        if self.current_index < len(self.prompts):
            return self.current_index
        else:
            return -1  # Fin du fichier

    def getNumberOfPrompts(self) -> int:
        """
        Renvoie le nombre total de prompts.

        Returns:
            int: Le nombre total de prompts dans le fichier CSV.
        """
        return len(self.prompts)


class JailbreakBenchJSON(DataSetController):
    def __init__(self, path_to_data_set: str):
        """
        Initializes the class by loading the JSON dataset file.

        Args:
            path_to_data_set (str): The path to the JSON dataset file.
        """
        self.current_index = 0
        # Load the JSON data from the file
        with open(path_to_data_set, "r") as f:
            self.data = json.load(f)

        # Extract the jailbreaks section which contains the prompts
        self.prompts = self.data.get("jailbreaks", [])

    def nextPrompt(self) -> str:
        """
        Returns the next prompt (question) from the JSON data.

        Returns:
            str: A string representing the next prompt.
        """
        if self.current_index < len(self.prompts):
            prompt = self.prompts[self.current_index]["prompt"]
            self.current_index += 1
            return str(prompt)
        else:
            return None  # Return None if there are no more prompts

    def getCurrentIndex(self) -> int:
        """
        Returns the current index.

        Returns:
            int: Current index, -1 if the end of the dataset is reached.
        """
        if self.current_index >= len(self.prompts):
            return -1
        return self.current_index

    def getNumberOfPrompts(self) -> int:
        """
        Returns the total number of prompts in the dataset.

        Returns:
            int: Total number of prompts.
        """
        return len(self.prompts)
