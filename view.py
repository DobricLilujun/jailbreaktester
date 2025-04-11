"""
File name: view.py
Author: Nathan Foucher 
Contact: nathan.foucher@ext.uni.lu
Created: 30/09/2024
Version: 1.1
Description: File to handle the applications GUI.
"""

from datetime import datetime
import sys
import webbrowser
import lib.controller as controller
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QMessageBox,
    QComboBox,
    QPushButton,
    QCheckBox,
    QTextEdit,
    QFileDialog,
    QHBoxLayout,
    QProgressBar,
    QMainWindow,
    QStatusBar,
    QTabWidget,
    QLineEdit,
)
from PyQt5.QtCore import Qt, pyqtSlot, QThread


class View(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM Jailbreak tester")
        self.initUI()

    @pyqtSlot(str)
    def update_current_question(self, data):
        self.text_question.setText(data)

    @pyqtSlot(str)
    def update_current_response(self, data):
        self.text_response.setText(data)

    @pyqtSlot(int)
    def update_progress(self, value: int):
        self.pbar.setValue(value)

    def initUI(self):
        # Création du layout principal
        layout = QVBoxLayout()

        # Liste déroulante pour sélectionner le LLM testé
        self.label_llm = QLabel("Select the LLM to test:")
        self.combo_llm = QComboBox()
        self.combo_llm.addItems(
            controller.LLM_MODELS.keys()
        )  # Ajouter les LLM disponibles ici

        # Liste déroulante pour sélectionner le type de dataset
        self.label_dataset = QLabel("Select the dataset type to use:")
        self.combo_dataset = QComboBox()
        self.combo_dataset.addItems(
            controller.DATA_SET_TYPE.keys()
        )  # Ajouter les LLM disponibles ici

        # Création des onglets
        self.tabs = QTabWidget()

        # Création du contenu pour l'onglet "LLM Test Campaign"
        self.llm_test_tab = QWidget()
        self.llm_test_layout = QVBoxLayout()
        self.llm_test_tab.setLayout(self.llm_test_layout)

        # Création du contenu pour l'onglet "Groundtruth Evaluation"
        self.groundtruth_tab = QWidget()
        self.groundtruth_layout = QVBoxLayout()
        self.groundtruth_tab.setLayout(self.groundtruth_layout)

        # Ajout des onglets au widget QTabWidget
        self.tabs.addTab(self.llm_test_tab, "LLM Test Campaign")
        self.tabs.addTab(self.groundtruth_tab, "Groundtruth Evaluation")

        # Bouton pour sélectionner le chemin du fichier groundTruth
        gt_file_path_layout = QHBoxLayout()
        self.groundtruth_explanation = QLabel(
            "This mode enable users to manually define the ground truth of already done test campaigns. Please choose the result json file of a test campaign made with this tool."
        )

        self.gt_button_path = QPushButton("Choose source...")
        self.gt_button_synchronize = QPushButton(
            "Synchronize ground truth from another file..."
        )
        self.gt_button_synchronize.clicked.connect(self.request_gt_sync)
        self.gt_button_synchronize.setEnabled(False)
        self.gt_label_path = QLabel("No source selected")
        self.groundtruth_layout.addWidget(self.groundtruth_explanation)
        gt_file_path_layout.addWidget(self.gt_button_path, 1, alignment=Qt.AlignLeft)
        gt_file_path_layout.addWidget(self.gt_label_path, 9, alignment=Qt.AlignLeft)
        gt_file_path_layout.addStretch()
        self.gt_button_path.clicked.connect(
            lambda: self.open_file_dialog(
                "Please choose a test campaign", "JSON Files (*.json)"
            )
        )

        self.groundtruth_layout.addLayout(gt_file_path_layout)
        self.groundtruth_layout.addWidget(self.gt_button_synchronize)

        # Bouton pour sélectionner le chemin du fichier
        file_path_layout = QHBoxLayout()
        self.button_path = QPushButton("Choose source...")
        self.label_path = QLabel("No source selected")
        file_path_layout.addWidget(self.button_path, 1, alignment=Qt.AlignLeft)
        file_path_layout.addWidget(self.label_path, 9, alignment=Qt.AlignLeft)
        file_path_layout.addStretch()
        self.button_path.clicked.connect(
            lambda: self.open_file_dialog("Please choose a dataset", "All Files (*)")
        )

        # Auto mode
        auto_mode_layout = QHBoxLayout()
        self.label_validator = QLabel("Select the classifier:")
        self.combo_validator = QComboBox()
        self.combo_validator.addItems(controller.CLASSIFIER_MODELS.keys())
        self.combo_validator.setEnabled(False)
        self.checkbox_auto_mode = QCheckBox("Auto mode")
        self.checkbox_auto_mode.clicked.connect(self.update_validator_combo)
        self.line_option_classifier = QLineEdit()
        self.line_option_classifier.setEnabled(False)
        self.line_option_classifier.textChanged.connect(self.generate_cli_command)
        auto_mode_layout.addWidget(self.checkbox_auto_mode, alignment=Qt.AlignLeft)
        auto_mode_layout.addWidget(self.label_validator, alignment=Qt.AlignLeft)
        auto_mode_layout.addWidget(self.combo_validator, alignment=Qt.AlignLeft)
        auto_mode_layout.addWidget(self.line_option_classifier)

        # Campaign name
        self.campaign_name_label = QLabel("Campaign name :")
        self.campaign_name = QLineEdit()
        self.campaign_name.setText(str(datetime.now().isoformat()))
        self.campaign_name_layout = QHBoxLayout()
        self.campaign_name_layout.addWidget(self.campaign_name_label)
        self.campaign_name_layout.addWidget(self.campaign_name)

        # CLI command
        self.label_cli_command = QLabel("CLI command :")
        self.cli_command_label = QTextEdit()
        self.cli_command_label.setMaximumHeight(50)
        self.cli_command_label.setReadOnly(True)
        self.cli_command_label.setEnabled(False)

        # Zone de texte pour afficher la question courante
        self.label_question = QLabel("Question asked :")
        self.text_question = QTextEdit()
        self.text_question.setReadOnly(True)  # La zone de texte est en lecture seule

        # Zone de texte pour afficher la réponse
        self.label_response = QLabel("LLM's response :")
        self.text_response = QTextEdit()
        self.text_response.setReadOnly(True)  # La zone de texte est en lecture seule

        # Boutons accepter ou non le prompt
        self.label_evaluation = QLabel("Evaluation :")
        evaluation_layout = QHBoxLayout()

        self.jailbreak_successfull_button = QPushButton("Successfully jailbroken")
        self.jailbreak_successfull_button.setProperty("value", True)
        self.jailbreak_successfull_button.setEnabled(False)
        self.jailbreak_successfull_button.clicked.connect(self.make_jailbreak_decision)

        self.jailbreak_failed_button = QPushButton("Jailbreak failed")
        self.jailbreak_failed_button.setProperty("value", False)
        self.jailbreak_failed_button.setEnabled(False)
        self.jailbreak_failed_button.clicked.connect(self.make_jailbreak_decision)

        evaluation_layout.addWidget(self.jailbreak_successfull_button)
        evaluation_layout.addWidget(self.jailbreak_failed_button)

        # Boutons demarrer et stopper
        start_stop_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_tester)
        self.start_button.clicked.connect(self.start_tester)
        start_stop_layout.addWidget(self.start_button)
        start_stop_layout.addWidget(self.stop_button)

        # Bouton pour afficher l'interface web
        self.show_results = QPushButton("Open results in browser")
        self.show_results.clicked.connect(self.open_web_interface)

        # Bar de progression
        self.pbar = QProgressBar(self)
        self.pbar.setMaximum(100)
        self.pbar.setMinimum(0)

        # Créer une barre d'état (foot bar)
        self.status_bar = QStatusBar()
        self.setStatusBar(
            self.status_bar
        )  # Associer la barre d'état à la fenêtre principale

        # Initialiser la barre d'état
        self.status_bar.showMessage(
            "Please choose a source file for the data set." "JSON Files (*.json)"
        )  # Message par défaut

        # Ajout des éléments au layout
        self.llm_test_layout.addWidget(self.label_llm)
        self.llm_test_layout.addWidget(self.combo_llm)
        self.llm_test_layout.addWidget(self.label_dataset)
        self.llm_test_layout.addWidget(self.combo_dataset)
        self.llm_test_layout.addLayout(file_path_layout)
        self.llm_test_layout.addLayout(auto_mode_layout)
        self.llm_test_layout.addLayout(self.campaign_name_layout)
        self.llm_test_layout.addWidget(self.label_cli_command)

        self.llm_test_layout.addWidget(self.cli_command_label)

        layout.addWidget(self.tabs)
        layout.addWidget(self.label_question)
        layout.addWidget(self.text_question)
        layout.addWidget(self.label_response)
        layout.addWidget(self.text_response)
        layout.addWidget(self.label_evaluation)
        layout.addLayout(evaluation_layout)
        layout.addWidget(self.pbar)
        layout.addLayout(start_stop_layout)
        layout.addWidget(self.show_results)

        # Last connector for cli command generation
        self.combo_llm.currentTextChanged.connect(self.generate_cli_command)
        self.combo_dataset.currentTextChanged.connect(self.generate_cli_command)
        self.combo_validator.currentTextChanged.connect(self.generate_cli_command)
        self.campaign_name.textChanged.connect(self.generate_cli_command)

        # Configuration de la fenêtre principale
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.setWindowTitle("Jailbreak Tester")
        self.resize(800, 600)

    def open_file_dialog(
        self, text="Please choose a dataset", file_type="All Files (*)"
    ):
        # Ouvre une boîte de dialogue pour sélectionner un fichier
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(
            self, text, "", file_type, options=options
        )
        if file:
            print(f"Fichier sélectionné: {file}")
            # Test campaign
            if self.tabs.currentIndex() == 0:
                self.label_path.setText(file)
                self.gt_button_synchronize.setEnabled(False)
                self.gt_label_path.setText("No source selected")
            # Ground truth
            elif self.tabs.currentIndex() == 1:
                self.gt_label_path.setText(file)
                self.gt_button_synchronize.setEnabled(True)
                self.label_path.setText("No source selected")

            self.generate_cli_command()
            self.start_button.setEnabled(True)
            self.status_bar.showMessage("Ready !")

    def start_tester(self):
        try:
            # Test campaign
            self.tabs.setEnabled(False)
            if self.tabs.currentIndex() == 0:
                llm = self.combo_llm.currentText()
                dataset_type = self.combo_dataset.currentText()
                source = self.label_path.text()
                auto_mode = self.checkbox_auto_mode.isChecked()
                validator = self.combo_validator.currentText()
                validator_options = self.line_option_classifier.text()
            # Ground truth
            elif self.tabs.currentIndex() == 1:
                llm = "GroundTruth"
                dataset_type = "GroundTruth"
                source = self.gt_label_path.text()
                auto_mode = False
                validator = None
                validator_options = None

            # Calling the controller
            self.tester_worker = controller.TesterWorker(
                self.campaign_name.text(),
                llm,
                dataset_type,
                source,
                auto_mode,
                validator,
                validator_options,
            )
        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("An error occured")
            msg.setText(str(e))
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            self.tabs.setEnabled(True)
            return

        # Connections
        self.tester_worker.finished.connect(self.leave_tester)
        self.tester_worker.question_update.connect(self.update_current_question)
        self.tester_worker.response_update.connect(self.update_current_response)
        self.tester_worker.request_decision.connect(
            self.enable_jailbreak_decision_buttons
        )
        self.tester_worker.progress_update.connect(self.update_progress)
        self.tester_worker.state_update.connect(self.status_bar.showMessage)

        # Thread creation
        self.thread = QThread()
        self.tester_worker.moveToThread(self.thread)
        self.thread.started.connect(self.tester_worker.startWorker)
        self.thread.start()

        # Interface update
        self.stop_button.setEnabled(True)
        self.start_button.setEnabled(False)
        self.jailbreak_failed_button.setEnabled(False)
        self.jailbreak_successfull_button.setEnabled(False)
        self.text_question.setText("")
        self.text_response.setText("")
        self.pbar.setValue(0)

    def stop_tester(self):
        self.tester_worker.stop()

    def show_message(self, msg: str):
        self.status_bar.showMessage(msg)

    def leave_tester(self, exit_code: int, log: str):

        # Leaving the thread
        self.tester_worker.deleteLater()
        self.thread.quit()

        # Check for errors
        if exit_code != 0:
            self.status_bar.showMessage("An error occured, ready")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("An error occured")
            msg.setText(str(log))
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

        # Interface update
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)
        self.jailbreak_failed_button.setEnabled(False)
        self.jailbreak_successfull_button.setEnabled(False)
        self.text_question.setText("")
        self.text_response.setText("")
        self.pbar.setValue(100)
        self.tabs.setEnabled(True)

        if self.checkbox_auto_mode.isChecked():
            self.combo_validator.setEnabled(True)
        else:
            self.combo_validator.setEnabled(False)

    def enable_jailbreak_decision_buttons(self):
        self.jailbreak_successfull_button.setEnabled(True)
        self.jailbreak_failed_button.setEnabled(True)

    def make_jailbreak_decision(self):
        value = self.sender().property("value")
        self.jailbreak_successfull_button.setEnabled(False)
        self.jailbreak_failed_button.setEnabled(False)
        self.tester_worker.continue_task(value)

    def update_validator_combo(self):
        if self.checkbox_auto_mode.isChecked():
            self.combo_validator.setEnabled(True)
            self.generate_cli_command()

        else:
            self.line_option_classifier.setText("")
            self.line_option_classifier.setEnabled(False)
            self.combo_validator.setEnabled(False)
            self.cli_command_label.setEnabled(False)
            self.cli_command_label.setText("")

    def open_web_interface(self):
        webbrowser.open("http://127.0.0.1:5000")

    def generate_cli_command(self):
        if self.checkbox_auto_mode.isChecked():
            validator = self.combo_validator.currentText()

            options = controller.CLASSIFIER_MODELS.get(validator).get("extra")

            if options != None :
                if self.line_option_classifier.text() == "" or self.sender() == self.combo_validator:
                    self.line_option_classifier.setText(options)
                    self.line_option_classifier.setEnabled(True)
                cli_classifier_options = (
                    f"--classifier_options '{self.line_option_classifier.text()}'"
                )

            else:
                self.line_option_classifier.setText("")
                self.line_option_classifier.setEnabled(False)
                cli_classifier_options = ""
            self.cli_command_label.setEnabled(True)
            self.cli_command_label.setText(
                f"python main.py --name '"
                + self.campaign_name.text()
                + "' --llm_model '"
                + self.combo_llm.currentText()
                + "' --data_set_type '"
                + self.combo_dataset.currentText()
                + "' --data_set_path '"
                + self.label_path.text()
                + "' --classifier '"
                + self.combo_validator.currentText()
                + "' "
                + str(cli_classifier_options)
            )
        else:
            self.cli_command_label.setEnabled(False)
            self.cli_command_label.setText("")
            self.line_option_classifier.setText("")
            self.line_option_classifier.setEnabled(False)

    def request_gt_sync(self):
        # Ouvre une boîte de dialogue pour sélectionner un fichier de campagne
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Please choose a test campaign with ground truth",
            "",
            "JSON Files (*.json)",
            options=options,
        )
        try:
            if file:
                print(f"Fichier sélectionné: {file}")
                controller.synchronize_ground_truth(file, self.gt_label_path.text())

        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("An error occured")
            msg.setText(str(e))
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            self.status_bar.showMessage("An error occured, ready")
            return

        self.status_bar.showMessage(
            f"Ground Truth successfully synchronized from {file} to {self.gt_label_path.text()} !"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    view = View()
    view.show()
    sys.exit(app.exec_())
