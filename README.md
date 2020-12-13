# Covid-19 Fact Checking
https://github.com/AvkashSharma/covid-19_fact_checking

# Team Members
- Karthikan Jeyabalan (40032932)
  - Training of NB-BOW
  - Trace file

- Nirusan Nadarajah (29600094)
  - Vocabulary (OV & FV)
  - Evaluation metric

- Avkash Sharma (40012077)
  - Predicting of NB-BOW
  - Run LSTM
  - Vocabulary (OV)

# Set-up 
- It is recommended to create a virtual environment using Conda. Host PC should have Anaconda or Mini conda installed
  1) Create a virtual environment with python 
      - ```conda create -n my_env python=3.8 ```
  2) Activate the environment
      - ```conda activate my_env```
      
- If Anaconda is not used then Download and install python version 3.8.0
  - Follow the instructions in the link to install python 3.8.0 https://www.python.org/downloads/release/python-380/
  
## Install Required Libraries
- pandas 
  - ```pip install pandas```
- numpy 
  - ```pip install numpy```
- tqdm 
  - ```pip install tqdm```
 
 
# Steps to run assignment
- Clone/Download the repo to the host machine
- Open terminal and cd into to project directory(covid-19_fact_checking)
- Run the nb_classifier by executing the following command in the terminal: 
  - ```python nb_classifier.py```
