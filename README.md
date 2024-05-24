Student Performance Prediction System

Project Overview

This project aims to predict student performance in secondary education using various machine learning models. It includes data preprocessing, model training, and a detailed analysis of the results to provide insights into factors influencing academic success.

Features

    Multiple regression models including Linear Regression, Random Forest, and LightGBM.
    Comprehensive data preprocessing functionalities.
    Performance evaluation and predictions stored for review.


This project uses a set of models to predict students' academic performance based on various factors. The project includes data preprocessing, model training, and prediction processes.


## File Structure

    helpers.py: A file containing functions for data preprocessing processes. It includes handling missing data, detecting and correcting outliers, and one-hot encoding.
    model.py: A file hosting prediction models (Linear Regression, Random Forest, LightGBM).
    model/: A folder that stores trained model objects in .pkl format.
    studentModel.py: The main Python script where data loading, model training, and prediction processes are performed.
    NOTES.txt: Contains model results, performance evaluations, and analytical comments.
    resources/: A folder containing the datasets and project details necessary for the project.


## Setup

To run the project locally, follow these steps:

    Install the required libraries:
    pip install -r requirements.txt

    Activate the virtual environment:
    source venv/bin/activate # For Linux or MacOS
    venv\Scripts\activate # For Windows

    Start the application:
    python main.py


Usage

The studentModel.py script automatically performs model training and prediction processes. To run the script:
python studentPredictive.py


Docker Usage
# To run the project in a container using Docker:

Build the Docker image:
	docker build -t student-prediction .

Start the Docker container:
	docker run -p 8000:8000 student-prediction


