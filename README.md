![UTA-DataScience-Logo](https://github.com/user-attachments/assets/36b0607e-06da-485c-97a1-34a4f0552141)

# California Wildfire Forecasting

* This repository presents a deep learning approach to forecasting wildfire occurrences in California using time series models (LSTM, GRU, Transformer, etc.), trained on weather and fire start data from [Zenodo Wildfire Dataset](https://zenodo.org/records/14712845). 

## OVERVIEW

  * **Background:** Wildfires in California pose a growing threat to both human life and the environment, often driven by complex interactions of weather patterns, seasonal changes, and past fire occurrences
  * **Project Goal:** The goal of this project is to build a forecasting tool based on a machine learning system that can predict the likelihood of a wildfire starting on a given day, using the prior 21 days of meteorological data and other time-dependent features
  * **Approach:** This task is a supervised binary classification task and is evaluated using several deep learning time series models (GRU, BiLSTM, Stacked LSTM, CNN+LSTM, Transformer). The approach emphasizes recall to minimize missed fire predictions, and threshold tuning and ROC-AUC/F1 evaluation was used to select the best model for deployment in a forecasting tool
  * **Summary of Performance** The best model, CNN+LSTM, achieved a recall of 0.85 and an F1-score of 0.67 on the test set, with an ROC-AUC of 0.79, effectively identifying potential wildfire days while balancing false alarms

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
  * **Bidirectional LSTM:** Allows the model to access both past and future context, improving sequence understanding. Learned that it's not the best for the data as there are no future sequences
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

The primary goal of this project is to accurately detect wildfire start days, where failing to identify a true fire (false negative) can have dangerous, even deadly consequences. For this reason, I prioritized ***Recall & ROC-AUC > F1 > Precision***:
* **Recall:** Catching as many actual fire events as possible is crucial to minimize missed alarms
* **F1:** Balances recall and precision, helping prevent over-alerting while still catching fires
* **Precision:** Excessive false positives can damage trust in the system and strain emergency resources
* **ROC-AUC:** Measures the model’s ability to distinguish between fire and non-fire days, regardless of threshold

<img width="566" alt="Screenshot 2025-07-01 at 11 45 12 PM" src="https://github.com/user-attachments/assets/2ee6337d-7585-44fd-b61e-3e54f9190dfb" />
<img width="709" alt="Screenshot 2025-07-01 at 11 35 18 PM" src="https://github.com/user-attachments/assets/31c2d41e-30f8-4711-b0ab-e4173f2e1333" />

While the GRU and Transformer models showed slightly better ROC-AUC and F1 scores, the ***CNN+LSTM architecture achieved the highest recall (0.85)***, which is critical in the context of wildfire forecasting where failing to detect a fire can have severe consequences. For this reason, ***CNN+LSTM was selected as the final model for the forecasting tool.*** Note that although XGBoost performed the best along with the Decision Tree model, the main goal was to choose the best **time-series** model. Those were just for comparison and curiosity.


### Conclusions

After selecting ***CNN + LSTM*** as the final model, I developed a wildfire forecasting prototype that evaluates ***random 21-day sequences***. For each sampled window, it predicts the likelihood of a fire on the 22nd day. The output includes the starting index of the window, the day being predicted, the model’s fire probability score, the predicted class (fire or no fire), the actual label from the dataset, and whether the prediction was correct (1) or incorrect (0). Below are random instances to showcase the model's performance:

<img width="371" alt="Screenshot 2025-07-02 at 12 37 56 AM" src="https://github.com/user-attachments/assets/85a289d6-b540-412f-8554-13871c536a05" />

Of the 10 randomly selected test cases, the model correctly predicted 7 out of 10 wildfire events. These results align with the final CNN+LSTM model’s evaluation metrics: Precision = 0.56, Recall = 0.85, F1-score = 0.67, and ROC-AUC = 0.7981. ***In the few incorrect cases, the model predicted a fire when none occurred (false positives), which is reflected in the precision and F1-score. However, it successfully identified all actual fire days in the sample, reinforcing the model’s strong recall and ability to minimize missed fire events. While this comes with a tradeoff of slightly more false alarms, the model performs well in prioritizing safety, and I’m proud of its ability to support early warning systems by capturing fire risks effectively.***

Tree-based models (especially XGBoost) achieved higher precision and ROC-AUC, meaning they were better at distinguishing between fire and no-fire days with fewer false positives. But they lacked the time window structure to fully leverage sequential temporal trends. ***Time series models, especially those using sliding windows like CNN + LSTM, were better at capturing short-term patterns and precursor signals, such as increasing wind speeds and shifting temperatures before a fire event.*** These models generally prioritized recall, catching more actual fires, which is ideal for early warning systems.

Overall, incorporating a time-windowed approach significantly improved the model’s ability to detect wildfire trends, reinforcing the importance of temporal context in predictive modeling of environmental events.

### Future Work

* **Incorporate Windowed Inputs into Tree-Based Models**
  * Explore how Random Forest and XGBoost perform when fed sliding window sequences similar to those used in deep learning models. This may improve their ability to capture temporal trends
* **Feature Engineering & Expansion**
  * Experiment with creating additional features or reintroducing previously dropped ones to better capture environmental dynamics that lead to fire events
* **Build an Interactive Forecasting Dashboard**
  * Develop a user-friendly dashboard based on the current prototype that accepts 21 days of recent environmental inputs and outputs a prediction: “Based on the data of the past 3 weeks, there is a __% chance there might be a fire on or around this day. Stay cautious!”
* **Integrate into Early Warning or Emergency Systems**
  * Expand this tool's application into real-time systems for supporting fire preparedness, emergency response, or public safety alerts

## HOW TO REPRODUCE RESULTS

### Overview of Files in Repository

The list below follows the chronological order in which each component of the project was developed:

* **Tabular Project Proposal MNC.pptx:** This project's proposal which includes background information and an abstract to the project
* **Tabular Project Proposal MNC.pdf:** PDF version of the Tabualr Project Proposal MNC.pptx powerpoint
* **CA_Weather_Fire_Dataset_1984-2025.csv:** Original data set downloaded from Zenodo
* **ProjectRoughDraft.ipynb:** My notebook before officially working on this project's feasibility and prototype. This was to help me understand my data through varioius in-depth visualizations and to play around with different models and come up with a plan to approach this problem
* **Feasibility_Tabular_MNC.ipynb:** This notebook includes EDA, visualization, and data pre-processing before running the models
* **firedf_cleaned.csv:** Output data frame after going through Feasibility_Tabular_MNC.ipynb pipeline 
* **Prototype_Tabular_MNC.ipynb:** Using firedf_cleaned.csv, this notebook trains the chosen models, chooses the best time series model, and includes the built wildfire forecasting tool prototype at the very end

### Software Setup

This project was developed and executed entirely in Jupyter Notebook. If you don’t already have it installed, you can download it as part of the Anaconda distribution or install it via pip "pip install notebook".

* **Data Handling & Visualization:**
  * pandas, numpy, matplotlib, seaborn
* **Preprocessing & Evaluation:**
  * sklearn.preprocessing:
    * MinMaxScaler, StandardScaler, OrdinalEncoder
  * sklearn.model_selection:
    * train_test_split
  * sklearn.metrics:
    * recall_score, precision_score, f1_score, roc_auc_score, roc_curve, classification_report, confusion_matrix
  * sklearn.utils:
    * class_weight
* **Machine Learning Models:**
  * sklearn.linear_model.LogisticRegression
  * sklearn.tree.DecisionTreeClassifier
  * sklearn.ensemble.RandomForestClassifier
* **Deep Learning (TensorFlow/Keras):**
  * tensorflow
    * Sequential, Model, Dense, Dropout, LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D, EarlyStopping, Recall, AUC
    * MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D (used for transformer model)
* **Gradient Boosting:**
  * xgboost

### Data

* **DATA DOWNLOAD LINK:** https://zenodo.org/records/14712845
* **All preprocessing and cleanup steps are documented and executed in the Feasibility_Tabular_MNC.ipynb notebook.** This includes:
  * Handling missing values based on distribution (mean, median, or zeros)
  * Cyclical encoding of seasonal and temporal features
  * Dropping redundant or highly correlated columns
  * Scaling features using MinMax normalization
  * Encoding the target variable
  * Sequence generation using a 21-day rolling window for time series modeling (in the Prototype_Tabular_MNC.ipynb notebook)

### Training

* Install required packages in notebook
* Download and prepare the data from Zenodo and the Feasibility_Tabular_MNC.ipynb pipeline, obtaining firedf_cleaned.csv
* Train models in this order: LSTM, GRU, BiLSTM, Stacked LSTM, CNN + LSTM, Transformer, Decision Tree, XGBoost
  * All models use Binary Crossentropy loss, Adam optimizer, and EarlyStopping callback with patience of 3 and val_loss monitoring
 * After training the base model, tune thresholds to balance Recall and F1 on the **validation set**
* Use the obtained threshold value on the **test set**

***For reference, see Prototype_Tabular_MNC.ipynb***

#### Performance Evaluation

* For each model:
  * Calculate key classification metrics: precision, recall, F1-score, ROC-AUC
  * Print classification report on validation set pre-tuned
  * Print classification report on validation set post-tuned
  * Print classification report on test set post-tuned
  * Plot ROC curves and compare models visually

***For reference, see Prototype_Tabular_MNC.ipynb***

## CITATIONS

[1] Abatzoglou, J. T., & Williams, A. P. (2018). Impact of anthropogenic climate change on wildfire across western US forests. *PNAS.*

[2] Cal Fire. “Incidents.” *CAL FIRE*, California Department of Forestry and Fire Protection, https://www.fire.ca.gov/incidents/. Accessed 28 June 2025.

[3] Keith, Michael. “Exploring the LSTM Neural Network Model for Time Series.” *Towards Data Science*, 13 Jan. 2023, https://towardsdatascience.com/exploring-the-lstm-neural-network-model-for-time-series-8b7685aa8cf/.

[4] Moore, Andrew. “Explainer: How Wildfires Start and Spread.” *College of Natural Resources News*, 3 Dec. 2021, https://cnr.ncsu.edu/news/2021/12/explainer-how-wildfires-start-and-spread/.

[5] Yavas, C. E., et al. California Weather and Fire Prediction Dataset (1984–2025) with Engineered Features. *Zenodo*, 21 Jan. 2025, doi:10.5281/zenodo.14712845. 

[6] Yuan, X., & Wu, X. (2022). Machine learning-based wildfire risk mapping in California using remote sensing. *ISPRS Journal of Photogrammetry and Remote Sensing.*
