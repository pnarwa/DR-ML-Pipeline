### Table of Contents

1. [Installation](#installation)
2. [Business Objectives](#objectives)
3. [File Descriptions](#files)
4. [Execution Instructions](#exec)
5. [Results](#results)
6. [Acknowledgements](#ack)

## Installation <a name="installation"></a>

The code has been developed using the Anaconda distribution of Python. The version of Python used is v3.9.7.
The main libraries used in developing the code are pandas, numpy, nltk, sqlalchemy, sklearn, skmultilearn, flask, plotly
The version of sklearn used in the implementation : 0.24.2
The version of skmultilearn used in the implementation : 0.2.0

Command to install the skmultilearn library:
pip install scikit-multilearn 

The links below have further details on installation of the skmultilearn library
- http://scikit.ml/
- https://pypi.org/project/scikit-multilearn/

## Business Objectives<a name="objectives"></a>
The objective here is to develop a machine learning pipeline which builds a classification model that can classify disaster response messages.
With the help of the webapp developed as part of the project, a user can input a message pertaining to a diasaster, and get a classification of the message into relevant disaster response categories.
The disaster response messages and categories used in the analysis are made available in collaboration with Figure Eight now [Appen](http://appen.com).

## File Descriptions <a name="files"></a>
The directories and files are organized as depicted below:
- \notebooks : notebooks used in preparing the ETL and ML pipelines
      - ETLPipelinePreparation.ipynb
      - MLPipelinePreparation.ipynb
- \data : data used in the analyis, and the ETL pipeline developed to process the data 
      - disaster_categories.csv
      - disaster_messages.csv
      - DisasterResponse.db
      - process_data.py
- \models : ML pipeline and the classification model built using the pipeline
      - train_classifier.py
      - model_classifier.pkl
- \app : flask program and templates related to the webapp
      - run.py
- \app\templates
      - master.html
      - go.html


## Execution Instructions <a name="exec"></a>
- To execute the ETL pipeline that cleans data and stores it in a sqlite database, run the below command:
  `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- To execute the ML pipeline that trains the classifier and saves the model, run the below command:
  `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
- To execute the webapp, cd to the app directory and run the below command:
  `python run.py`

## Results<a name="results"></a>
The URL to the GitHub repository is available [here](https://github.com/pnarwa/DR-ML-Pipeline)<br />

## Acknowledgements<a name="ack"></a>
- Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011). On the stratification of multi-label data. Machine Learning and Knowledge Discovery in Databases, 145-158. http://lpis.csd.auth.gr/publications/sechidis-ecmlpkdd-2011.pdf
- Piotr Szymański, Tomasz Kajdanowicz ; Proceedings of the First International Workshop on Learning with Imbalanced Domains: Theory and Applications, PMLR 74:22-35, 2017. http://proceedings.mlr.press/v74/szyma%C5%84ski17a.html
- The disaster response categories and messages used in the analysis are made available in collaboration with Figure Eight now [Appen](http://appen.com) 
