"""
File name: main.py
Author: Nathan Foucher 
Contact: nathan.foucher@ext.uni.lu
Created: 30/09/2024
Version: 1.1
Description: Main file used to run Jailbreak Tester. 
"""

import sys
import threading
from PyQt5.QtWidgets import QApplication
from view import View
from web import run_flask
import argparse
from lib.controller import TesterWorker
from linker import LLM_MODELS, DATA_SET_TYPE, CLASSIFIER_MODELS

def validate_campaign_name(camaign_name):
    if camaign_name == "" :
        raise ValueError("No campaign name setted.")

def validate_llm_model(llm_model):
    if llm_model not in LLM_MODELS:
        raise ValueError(
            f"Invalid LLM model '{llm_model}'.\n Available options are: {', '.join(LLM_MODELS.keys())}"
        )


def validate_data_set_type(data_set_type):
    if data_set_type not in DATA_SET_TYPE:
        raise ValueError(
            f"Invalid data set type '{data_set_type}'.\n Available options are: {', '.join(DATA_SET_TYPE.keys())}"
        )


def validate_classifier(classifier):
    if classifier not in CLASSIFIER_MODELS:
        raise ValueError(
            f"Invalid classifier model '{classifier}'.\n Available options are: {', '.join(CLASSIFIER_MODELS.keys())}"
        )


if __name__ == "__main__":

    # Argument parser to parse the CLI arguments
    parser = argparse.ArgumentParser(description="LLM Model Controller for Testing")

    parser.add_argument(
        "--name",
        type=str,
        required=False,
        help=f"Campaign name.",
    )

    parser.add_argument(
        "--llm_model",
        type=str,
        required=False,
        help=f"Name of the LLM model to use. Available options are: {', '.join(LLM_MODELS.keys())}",
    )
    parser.add_argument(
        "--data_set_type",
        type=str,
        required=False,
        help=f"Type of the dataset. Available options are: {', '.join(DATA_SET_TYPE.keys())}",
    )
    parser.add_argument(
        "--data_set_path", type=str, required=False, help="Path to the dataset file"
    )
    parser.add_argument(
        "--classifier",
        type=str,
        required=False,
        help=f"Classifier model to use (if in auto mode)Available options are: {', '.join(CLASSIFIER_MODELS.keys())}",
    )

    parser.add_argument(
        "--classifier_options",
        type=str,
        required=False,
        help="Options for the classifier. Please check your classifier documentation to get them. The default options are in linker.py",
    )

    args = parser.parse_args()

    if args.llm_model != None:

        print("Starting CLI mode...")

        try:
            # Validate campaign name
            validate_campaign_name(args.name)
            
            # Validate the LLM model
            validate_llm_model(args.llm_model)

            # Validate the data set type
            validate_data_set_type(args.data_set_type)

            # Validate the classifier
            validate_classifier(args.classifier)
            
            # If no options use the default ones in linker
            if args.classifier_options == None : 
                classifier_options = CLASSIFIER_MODELS.get(args.classifier).get('extra')
            else :
                classifier_options = args.classifier_options

        except ValueError as e:
            print(f"Error: {e}")
            exit(1)

        # Initialize the worker with the provided arguments
        print("Initializing worker...")
        try:
            worker = TesterWorker(
                campaign_name=args.name,
                llm_model=args.llm_model,
                data_set_type=args.data_set_type,
                data_set_path=args.data_set_path,
                auto_mode=True,
                classifier=args.classifier,
                classifier_options=classifier_options,
                cli_mode=True,
            )
            print(
                "Running, this might take a long time... To see the progress in real time please use the GUI."
            )
            # Start the worker process
            worker.startWorker()
        except Exception as e:
            print("An error occured :", str(e))
            exit(1)
        print("Done! Results are saved in the output directory as .json file.")
        exit(0)

    else:
        # Start the GUI mode
        print("Starting GUI mode...")
        # Start the Flask server in a separate thread
        try:
            flask_thread = threading.Thread(target=run_flask, daemon=True)
            flask_thread.start()
        except:
            print("Error while starting web server")

        app = QApplication(sys.argv)
        view = View()
        view.show()
        sys.exit(app.exec_())
