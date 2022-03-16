### Table of Contents

1. [Installation](#installation)
2. [Business Objectives](#objectives)
3. [File Descriptions](#files)
4. [Execution Instructions](#exec)
5. [Results](#results)
6. [Acknowledgements](#ack)

## Installation <a name="installation"></a>

The code has been developed using the Anaconda distribution of Python. The version of Python used is v3.9.7.
The main libraries used are pandas, numpy, nltk, sqlalchemy, sklearn, skmultilearn, flask, plotly.<BR>
The version of sklearn used in the implementation : 0.24.2<BR>
The version of skmultilearn used in the implementation : 0.2.0<BR>

Command to install the skmultilearn library:<BR>
pip install scikit-multilearn 

The links below have further details on installation of the skmultilearn library
- http://scikit.ml/
- https://pypi.org/project/scikit-multilearn/

As the dataset is a multi-labeled dataset and is highly imbalanced, the iterative_train_test_split method from the skmultilearn library is used to perform a iteratively stratified train/test split of the multi-labeled data.<BR>
Also, the IterativeStratification class from skmultilearn is used to perform a stratified sampling when creating the k-folds for cross-validation.

## Business Objectives<a name="objectives"></a>
The objective here is to develop a NLP(nautral language processing) based machine learning pipeline to build a classification model for classifying disaster response messages.<BR>
Using the webapp developed as part of the project, a user can input a message pertaining to a disaster, and get a classification of the message into relevant disaster response categories.<BR>
The disaster response messages and categories used in the analysis are made available in collaboration with Figure Eight now [Appen](http://appen.com).

## File Descriptions <a name="files"></a>
The directories and files are organized as depicted below:<BR>
- \notebooks : notebooks used in preparing the ETL and ML pipelines<BR>
      - ETLPipelinePreparation.ipynb<BR>
      - MLPipelinePreparation.ipynb<BR>
- \data : data used in the analysis, ETL pipeline developed to process the data<BR>
      - disaster_categories.csv<BR>
      - disaster_messages.csv<BR>
      - DisasterResponse.db<BR>
      - process_data.py<BR>
- \models : ML pipeline and the classification model built using the pipeline<BR>
      - train_classifier.py<BR>
      - model_classifier.pkl<BR>
- \app : flask program and templates related to the webapp<BR>
      - run.py<BR>
- \app\templates<BR>
      - master.html<BR>
      - go.html<BR>
- \screenshots : screenshots of the webapp home page and sample results pages
      


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
