## General ML test task results

This sub-folder has the next structure:
 - artifacts (folder with files we are getting after script runs)
 - constants.py (python file with config value we want to keep separate)
 - predict.py (python file to run prediction on trained model)
 - train.py (python file to run training of the model)
 - EDA.ipynb (Jupyter Notebook with exploratory data analysis on training features and label)
 
### How to run:

Two scripts were developed for this task:
 - train.py
 - test.py

They have no CLI argument so the execution is simple (from prepared venv) but the order is crucial.
constants.py file has every filename we need to run those so no need for the manual arguments input.

Run examples:

    python train.py

    python predict.py
