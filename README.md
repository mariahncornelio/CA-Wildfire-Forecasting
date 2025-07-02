![](UTA-DataScience-Logo.png)

# California Wildfire Forecasting

* This repository presents a deep learning approach to forecasting wildfire occurrences in California using time series models (LSTM, GRU, Transformer, etc.), trained on weather and fire start data from [Zenodo Wildfire Dataset](https://zenodo.org/records/14712845). 

## OVERVIEW

  * **Background:** Wildfires in California pose a growing threat to both human life and the environment, often driven by complex interactions of weather patterns, seasonal changes, and past fire occurrences.
  * **Project Goal:** The goal of this project is to build a forecasting tool based on a machine learning system that can predict the likelihood of a wildfire starting on a given day, using the prior 21 days of meteorological data and other time-dependent features.
  * **Approach:** This task is a supervised binary classification task and is evaluated using several deep learning time series models (GRU, BiLSTM, Stacked LSTM, CNN+LSTM, Transformer). Our approach emphasizes recall to minimize missed fire predictions, and we use threshold tuning and ROC-AUC/F1 evaluation to select the best model for deployment in a forecasting tool.
  * **Summary of Performance** The best model, CNN+LSTM, achieved a recall of 0.85 and an F1-score of 0.67 on the test set, with an ROC-AUC of 0.79, effectively identifying potential wildfire days while balancing false alarms.

## SUMMARY OF WORK DONE

### Data

  * **Type:**
    * Input: Daily weather and temporal features in a CSV file (precipitation, temperature, wind speed, etc.)
    * Output: Binary flag indicating wildfire ignition on a given day (FIRE_START_DAY)
  * **Size:**
    * 14,988 rows of daily records from 1984-2025, 14 features
  * **Instances:**
    * After creating 21-day window, 10476 training samples, 2245 validation, and 2246 test samples. 70-15-15 split, respectively

#### Preprocessing / Clean up

* **Missing Values:**
  * PRECIPITATION values were imputed with 0 (reflecting realistic dry periods)
  * Normally distributed features (MAX_TEMP, MIN_TEMP, AVG_WIND_SPEED, WIND_TEMP_RATIO) were imputed with the mean
  * Right-skewed TEMP_RANGE was imputed using the median
* **Encoding:**
  * The target column FIRE_START_DAY was encoded as a binary integer
  * SEASON and DAY_OF_YEAR were encoded using cyclical features (sin/cos) to preserve periodicity
* **Datetime and Sorting:**
  * Dates were parsed using pd.to_datetime() and sorted chronologically to maintain sequence order
* **Feature Selection:**
  * Redundant or highly collinear features (e.g., WIND_TEMP_RATIO, MONTH, DATE) were dropped based on correlation analysis (threshold was r >= 0.9)
* **Normalization:**
  * A MinMaxScaler was applied to all numerical features except for cyclical encodings and the target
* **Other:**
  * Duplicate rows were removed
  * A sliding window function was implemented to generate 21-day sequences for time series models
  * Outliers were kept to preserve real-world variability in wildfire patterns

#### Data Visualization

<img width="691" alt="Screenshot 2025-07-01 at 6 35 27 PM" src="https://github.com/user-attachments/assets/fab01205-f919-4985-ac5e-608b928f0d46" />

EDA revealed several patterns associated with wildfire occurrences. Fires can start ***regardless of rainfall***, though they are ***more common when there is no precipitation*** or lagged precipitation, indicating dry conditions. ***Higher maximum and minimum temperatures***, especially above 55°F, and ***wider temperature ranges*** tend to correlate with fire starts. Wind plays a crucial role, as ***fires often occur on windier days***, both in real-time and lagged measures. A clear upward trend in fire frequency was observed over the years. Seasonality also emerged, with most fires ***occurring between June and August***, aligning with ***dry summer periods***. Additionally, interactions between wind and heat, reflected in the wind–temperature ratio, suggest ***vegetation dryness increases fire risk***.

<img width="547" alt="Screenshot 2025-07-01 at 6 38 59 PM" src="https://github.com/user-attachments/assets/a1354527-0fc2-4dd0-8c12-1a1102eeba7d" />

Finally, the dataset shows a ***moderate class imbalance***, with approximately 66% of samples labeled as no fire and 33% as fire, which was considered during model training.

![download](https://github.com/user-attachments/assets/2b072b8a-c4eb-4b74-94f0-6f25c0da780b)

The plot above shows feature trends during the 21 days leading up to a fire event on April 30, 2011. An important trend observed is that ***1–2 weeks before a fire starts, temperatures tend to drop while wind speeds increase***. This pattern likely reflects the ***arrival of colder, drier air, which can dry out vegetation*** and create conditions more favorable for wildfires.

### Problem Formulation

* **Input/Output**
  * Input: 21-day sequence of meteorological and seasonal features (such as temperature, wind, precipitation, etc.)
  * Output: Binary label indicating whether or not a fire would start on the 22nd day
* **Models Used:**
  * **LSTM:** Designed to handle sequential data, making it a natural fit for time series prediction
  * **GRU:** Faster alternative to LSTM, yields comparable performance with fewer parameters
  * **Bidirectional LSTM:** Allows the model to access both past and future context, improving sequence understanding. Learned that it's not the best for our data as there are no future sequences
  * **Stacked LSTM:** Multiple LSTM layers stacked to learn more abstract temporal patterns
  * **CNN + LSTM:** Combines convolution neural networks for local pattern extraction (days) and LSTM for temporal modeling
  * **Transformer:** Capable of capturing long-range dependencies and parallelizing training efficiently
  * **Random Forest & XGBoost:** Included to compare/benchmark against decision tree–based models. These did not use sliding windows but provided a comparison between deep learning time series models and traditional ensemble methods
* **Loss Function & Optimizer**
  * All models were trained with binary crossentropy and the Adam optimizer
  * Performance was monitored using metrics such as recall and ROC-AUC
* **Hyperparamters & Threshold Tuning**
  * Class weights applied to all models to handle class imbalance
  * Early stopping implemented to all models to prevent overfitting
  * Threshold tuning was used to prioritize recall, followed by F1-score, with optimal thresholds typically falling between 0.45 and 0.55, depending on the model

### Training

Model training was performed in **Jupyter Notebook** on a **MacBook Pro with an Apple M1 chip and 8 GB of memory.** The following libraries and frameworks were used throughout the training process:
* TensorFlow/Keras for building and training deep learning models (LSTM, GRU, BiLSTM, Stacked LSTM, CNN+LSTM, Transformer)
* scikit-learn for preprocessing, metrics, and tree-based models like Random Forest
* XGBoost for training a gradient-boosted decision tree classifier

Each model took **between 10 seconds to 1 minute to train** with Transformer taking the longest. In total, model training and evaluation spanned approximately **12 hours over 3 days**, including experimentation and threshold tuning. **Early stopping** was implemented for all deep learning models, with **validation loss as the monitored metric and a patience of 3 epochs, restoring the best weights** to avoid overfitting. Example below:

<img width="706" alt="Screenshot 2025-07-01 at 11 25 14 PM" src="https://github.com/user-attachments/assets/d203db20-529f-4500-9b01-f6c204394338" />

**Challenges:** Some challenges I had was overfitting within the models, but that was solved by early stopping, I learned. A key mistake I made during development involved tuning thresholds on the test set instead of the validation set, introducing bias into the model. I learned my mistake the hard way when I realized my scores were suspiciously too good to be true and corrected by reverting to proper validation-based tuning.

### Performance Comparison

The primary goal of this project is to accurately detect wildfire start days, where failing to identify a true fire (false negative) can have dangerous, even deadly consequences. For this reason, we prioritized ***Recall & ROC-AUC > F1 > Precision***:
* **Recall:** Catching as many actual fire events as possible is crucial to minimize missed alarms
* **F1:** Balances recall and precision, helping prevent over-alerting while still catching fires
* **Precision:** Excessive false positives can damage trust in the system and strain emergency resources
* **ROC-AUC:** Measures the model’s ability to distinguish between fire and non-fire days, regardless of threshold

<img width="566" alt="Screenshot 2025-07-01 at 11 45 12 PM" src="https://github.com/user-attachments/assets/2ee6337d-7585-44fd-b61e-3e54f9190dfb" />
<img width="709" alt="Screenshot 2025-07-01 at 11 35 18 PM" src="https://github.com/user-attachments/assets/31c2d41e-30f8-4711-b0ab-e4173f2e1333" />

While the GRU and Transformer models showed slightly better ROC-AUC and F1 scores, the ***CNN+LSTM architecture achieved the highest recall (0.85)***, which is critical in the context of wildfire forecasting where failing to detect a fire can have severe consequences. For this reason, ***CNN+LSTM was selected as the final model for the forecasting tool.*** Note that although XGBoost performed the best along with the Decision Tree model, the main goal was to choose the best **time-series** model. Those were just for comparison and curiosity.


### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## HOW TO REPRODUCE RESULTS

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


## CITATIONS

* Provide any references.
