# Jailbreak Tester
A simple tool to test prompts across multiple LLMs and evaluate the success of jailbreaking. The tool is designed to allow users to easily develop interfaces between their chosen LLMs, datasets, and classifiers.

## Presentation 
While working on LLM jailbreaking, our team found it difficult to easily compare different LLMs, prompt datasets, and classifiers. JailBreakTester was developed to address this challenge, providing users with a unified interface, standardized communication methods, and shared result files.

## Use cases 

### I. Perform a test campaign by hand
The primary use of the tool is to run test campaigns on multiple LLMs using different datasets.

![Use Case 1 Image](images/usecase1.png)


The user can select one of the LLMs they have implemented (see instructions below), a dataset type, and a classifier if "Auto mode" is enabled.

Without Auto mode enabled, the user will be required to manually determine whether each LLM response is successfully jailbroken. Using the manual mode the 'jailbreak_groundtruth' key in campaigns json will be set directly. 

### II. Perform a test campaign with a classifier
Users can also utilize their own implemented classifier. In this mode, the tool will automatically determine whether the jailbreak is successful based on the classifier's evaluation.

![Use Case 2 Image](images/usecase2.png)

If the classifier requires options, they can be entered in the text box on the right.

### III. Ground truth evaluation
When running a test campaign with a classifier, it can be useful to manually evaluate the success of jailbreaks afterward. A "Ground Truth Evaluation" tab is available for this purpose. Once the ground truth is evaluated, metrics such as precision, recall, F1 score, and the confusion matrix will be available.

Additionally, if you want to synchronize the ground truth between two test campaigns to avoid re-evaluating manually, a button is provided for that in the same tab. Using this the tool will add the groud truth values in the source file into the processsed file. 

When evaluating ground truth or synchronizing them, a new boolean key "jailbreak_groundtruth" is added in the campaigm json file processed. 

### IV. Results 
All test campaign results are stored as JSON files in the results directory. Results are accessible through a web interface, where you can sort campaigns by model, classifier, and dataset. Graphs are generated for all campaigns selected in the web interface.

### Shape of a campaigns file 

The structure of campaign files is the following : 
```json
{
    "name": "campaign_name",
    "model": "huggingface/meta-llama/Llama-2-7b-chat-hf",
    "data_set_path": "~/Developpement/jailbreaktester/dataset/data.json",
    "data_set_type": "SimpleJsonDataSet",
    "auto_mode": true,
    "classifier": "StringClassifier",
    "classifier_options": "",
    "timestamp": "2024-11-14T15:26:53.062067",
    "requests": [
        {
            "question": "Question asked to the llm",
            "response": "llm's response",
            "jailbreak_successful": "[bool] (decision of the classifier used)",
            "timestamp": "2024-11-14T15:26:59.689434",
            "jailbreak_groundtruth": "[bool] (decision of hand verification if done)"
        }
    ]
}
```


### Web interface
The web intergface enable users to see some metrics such as ASR, f1 score, precision, recall... Be aware that for some metrics the groudtruth has to be evaluated by the user. 

![Use Case 4 Image](images/usecase4.png)

You can export the metrics by using the export button located at the bottom of the page.

![Use Case 4 Bis Image](images/usecase4bis.png)

## How to install
```console
conda env create -f environment.yml
conda activate JailBreakTester
```
If the installation fails during llama_cpp building please use these commands and install again.
```console
conda env remove -n JailBreakTester
sudo apt install gcc
sudo apt install g++
export CMAKE_ARGS="-DLLAVA_BUILD=OFF"
```

If the launch of the application fails due to a PyQt error please try the following command and run again.
```console
sudo apt-get install libxcb-xinerama0
```

#### API key
If you want to use models hosted on OPEN AI or HF you will need to add an API key for each of theses services. To do so you need to create a '.env' file on the root of the project with the following shape : 

```console
HUGGINGFACE_API_KEY = "hf_xxxxxxxxxxxxxxxxxx"
OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxx"
```

## How to launch 

### The full application 
```console
python main.py
```
Several instances of the application can be used at the same time to perform multiple test campaigns.

### Only the GUI to test LLMs
```console
python view.py
```

### Only web interface to show results 
```console
python web.py
```

### Use the application with the cli
The application is designed to be used primarily through the terminal if needed. In the application’s GUI, when configuring a test campaign, the command that can be used in the CLI is displayed to the user.

Example : 
```console
python main.py --llm_model 'huggingface/meta-llama/Llama-2-7b-chat-hf' --data_set_type 'SimpleJsonDataSet' --data_set_path '/home/linux/Developpement/jailbreaktester/dataset/data.json' --classifier 'LlamaGuard3JailbreakJudge' 
```

### Application architecture 

## UML diagram 
![UML diagram](images/UML_diagram.svg)

## How to add your own class
This application is designed to allow users to utilize their own LLMs (either local or remote via API), datasets, and classifiers. To add your own assets, three abstract classes define the methods required for the application to function properly.

### lib.DataSetController
You define in this file  your own class to handle your datasets. Your class has to implement the abstract methods defined at the begining of the module.

```python
class DataSetInterface(ABC):

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

# You need to declare the implentation of 'DataSetInterface' in your class interface between the brackets :
class ImplentationDataSetExample(DataSetInterface):
```

### lib.LLMController
You define in this file your own class to handle your LLMs. Your class has to implement the abstract methods defined at the begining of the module.

```python
class LLMController(ABC):
    """
    Abstract class defining the interface to interact with a LLM model.
    """

    @abstractmethod
    def __init__(self, api_key: str = None, extra=None):
        """
        Initializes the class with the API key if needed.

        Args:
            api_key (str): The API key.
            extra (str): extra input maybe used by the controller.
        """
        pass

    @abstractmethod
    def askPrompt(self, prompt: str) -> str:
        """
        Abstract method that must be implemented to ask a question (prompt) to a language model.

        Args:
            prompt (str): The question to ask the LLM model.

        Returns:
            str: The response generated by the LLM model.
        """
        pass

# You need to declare the implentation of 'LLMController' in your class interface between the brackets :
class ImplentationLLM(LLMController):
```

### lib.ClassifierController
You define in this file your own class to handle your classifiers. Your class has to implement the abstract methods defined at the begining of the module.

```python
class BaseClassifier(abc.ABC):

    @abc.abstractmethod
    def __init__(
        self, api_key: str = None, extra: str = None, llm: LLMController = None
    ):
        """
        Initializes the class with the API key, LLMController or extra options if needed.

        Args:
            api_key (str): The API key if needed.
            extra (str): Extra input maybe used by the controller if needed.
            llm (LLMController): LLMController object used to communicate with the LLM choosed by the user if needed.
        """
        pass

    @abc.abstractmethod
    def classify_responses(
        self, prompts: list[str], responses: list[str]
    ) -> list[bool]:
        """
        Used to classify responses. Warning: at the current state of development only one prompt and response will be send to the classifier at the time.

        args:
            prompts (list[str]): list of prompts
            responses (list[str]): list of responses
        returns:
            list[bool]: True if the response is classified as jailbroken and False otherwise
        """
        pass

# You need to declare the implentation of 'BaseClassifier' in your class interface between the brackets :
class ImplentationClassifier(BaseClassifier):
```


## How to import my new class in the application ?
The file 'linker.py' is used to link your new class with the application's GUI and worker. Here is an example : 

```python
LLM_MODELS = {
    "huggingface/meta-llama/Llama-2-7b-chat-hf": {
        "local": False,
        "API_key": HF_API_KEY,
        "class": LLMController.HF,
        "extra": "meta-llama/Llama-2-7b-chat-hf",
    },
    "[LOCAL] Quantized-llama2-7b-chat": {
        "local": True,
        "class": LLMController.quantized_llama2_7b,
        "extra": "/home/linux/models/llama-2-7b-chat.Q4_K_M.gguf",
    },
}

DATA_SET_TYPE = {
    "SimpleJsonDataSet": {"class": DataSetController.SimpleJsonDataSet},
}

CLASSIFIER_MODELS = {
    "StringClassifier": {"class": ClassifierController.StringClassifier},
    "LlamaGuard3JailbreakJudge": {
        "class": ClassifierController.LlamaGuard3JailbreakJudge,
        "API_KEY": HF_API_KEY,
    },
    "Pert2Detect": {
        "class": ClassifierController.Pert2Detect,
        "API_KEY": HF_API_KEY,
        "extra": '--threshold 0.15 --smoothllm_num_copies 2 --smoothllm_pert_types ["RandomSwapPerturbation"] --smoothllm_pert_pct_min 5 --smoothllm_pert_pct_max 10',
    },
}

```
The 'API_key' and 'extra' inputs are not mandatory, depending on your LLM used. 

# How to use Jailguard 
Jailguard is a classifier made by a another team (https://github.com/shiningrain/JailGuard/tree/main). Their tool has been modified to be used by ours. 

This is experimental and not user friendly AT ALL. But, if you are willing to use the jailguard classifier please follow these steps. 

Go to ./jailguard then enter the following comand : 

```console
conda create -n JailGuard python=3.9.18
conda activate JailGuard
pip install -r requirements.txt
pip install matplotlib
pyhton -m spacy download en_core_web_md
```

This will create an environement used by jailbreaktester to communicate with JailGuard. 

It is possible that some libraries are missing, the error logs will tell which ones. In that case, open a terminal, activate JailGuard environment and install the libraries or tools requested.  

