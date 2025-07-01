![](UTA-DataScience-Logo.png)

# California Wildfire Forecasting

* This repository presents a deep learning approach to forecasting wildfire occurrences in California using time series models (LSTM, GRU, Transformer, etc.), trained on weather and fire start data from [Zenodo Wildfire Dataset](https://zenodo.org/records/14712845). 

## Overview

  * **Background:** Wildfires in California pose a growing threat to both human life and the environment, often driven by complex interactions of weather patterns, seasonal changes, and past fire occurrences.
  * **Project Goal:** The goal of this project is to build a forecasting tool based on a machine learning system that can predict the likelihood of a wildfire starting on a given day, using the prior 21 days of meteorological data and other time-dependent features.
  * **Approach:** This task is a supervised binary classification task and is evaluated using several deep learning time series models (GRU, BiLSTM, Stacked LSTM, CNN+LSTM, Transformer). Our approach emphasizes recall to minimize missed fire predictions, and we use threshold tuning and ROC-AUC/F1 evaluation to select the best model for deployment in a forecasting tool.
  * **Summary of Performance** The best model, CNN+LSTM, achieved a recall of 0.84 and an F1-score of 0.67 on the test set, with an ROC-AUC of 0.79, effectively identifying potential wildfire days while balancing false alarms.

## Summary of Work Done

Include only the sections that are relevant an appropriate.

### Data

* Data:
  * Type: For example
    * Input: medical images (1000x1000 pixel jpegs), CSV file: image filename -> diagnosis
    * Input: CSV file of features, output: signal/background flag in 1st column.
  * Size: How much data?
  * Instances (Train, Test, Validation Split): how many data points? Ex: 1000 patients for training, 200 for testing, none for validation

#### Preprocessing / Clean up

* Describe any manipulations you performed to the data.

#### Data Visualization

Show a few visualization of the data and say a few words about what you see.

### Problem Formulation

* Define:
  * Input / Output
  * Models
    * Describe the different models you tried and why.
  * Loss, Optimizer, other Hyperparameters.

### Training

* Describe the training:
  * How you trained: software and hardware.
  * How did training take.
  * Training curves (loss vs epoch for test/train).
  * How did you decide to stop training.
  * Any difficulties? How did you resolve them?

### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: various functions that are used in cleaning and visualizing data.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * models.py: Contains functions that build the various models.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

* Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Provide any references.
