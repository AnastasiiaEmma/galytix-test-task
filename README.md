# My Python App

This application processes a set of phrases by calculating semantic distances between them using pretrained Word2Vec vectors. The application includes functionality to calculate distances in batch mode and find the closest matching phrase to a user-input phrase in real-time.

## Directory Structure` 

```
galytix_test_task/ 
├──  **init**.py 
├── .gitignore 
├── init_pipeline.py 
├── logging.conf 
├── phrases.csv 
├── process_data.py 
├── README.md 
├── requirements.txt 
├── utils.py
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- Virtual environment (venv)

### Step-by-Step Guide

1. **Clone the Repository**
    ```
    git clone https://github.com/AnastasiiaEmma/galytix-test-task.git
    cd my_python_app
    ```

2.  **Create and Activate Virtual Environment**
    ```
    python3 -m venv venv
    source venv/bin/activate
    ```
    
4.  **Install Required Libraries**
    ```
    pip install -r requirements.txt 
    ```
    
5.  **Download Pretrained Word2Vec Vectors**
    
    - Download the pretrained Word2Vec vectors from  [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM)
    - Place the  `GoogleNews-vectors-negative300.bin.gz`  file in the project directory.

## Usage

### Initializing the Pipeline

1.  **Run  `init_pipeline.py`  to Load and Save Word2Vec Vectors**

    `python init_pipeline.py` 
    
    This script will load the first million vectors from the pretrained Word2Vec model and save them to  `vectors.csv`.
    

### Processing Data

1.  **Ensure  `phrases.csv`  is Present**
    
    The  `phrases.csv`  file should contain a single column named  `Phrases`  with the phrases to be processed.
    
2.  **Run  `process_data.py`  to Calculate Distances**

    `python process_data.py` 
    
    This script will:
    
    -   Load the phrases from  `phrases.csv`.
    -   Assign Word2Vec embeddings to each phrase.
    -   Calculate the cosine distances between the phrase embeddings.
    -   Save the distance matrix to  `phrase_distances.csv`.

### Finding the Closest Matching Phrase

The  `process_data.py`  script also includes functionality to find the closest matching phrase to a user-input phrase. To run the script, execute it from the terminal. When prompted `Please enter a phrase: ` , enter a phrase, and the script will display the closest matching phrase from the dataset.


## Logging and Error Handling

The application uses a logging configuration specified in  `logging.conf`. Ensure this file is present in the project directory to enable proper logging.

## Utility Functions

The  `utils.py`  file contains utility functions that assist with various tasks in the application. You can add more helper functions as needed.

## Contributing

Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.
### Explanation of Changes
- Added instructions to download the pretrained Word2Vec vectors and place the `GoogleNews-vectors-negative300.bin.gz` file in the project directory.
- Removed the `GoogleNews-vectors-negative300.bin.gz`, `vectors.csv`, and `phrase_distances.csv` from the directory structure since they are no longer included in the repository.
