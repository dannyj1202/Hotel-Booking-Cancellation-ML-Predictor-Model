# 🏨 Hotel Booking Cancellation ML Predictor

This project is a supervised machine learning solution built for **XYZ Hotels Group** to predict whether a hotel booking will be cancelled. By accurately forecasting cancellations, the hotel chain can proactively manage their inventory, reduce revenue loss, and improve operational efficiency.

---

## 🔍 Problem Statement

XYZ Hotels Group suffers from revenue and inventory loss due to frequent **booking cancellations**. This model aims to **predict whether a booking will be cancelled**, allowing the hotel to take corrective actions (like offering discounts or overbooking compensation) in advance.

---

## 🧠 Solution Overview

- ✅ **Type**: Binary Classification
- ✅ **Model**: Decision Tree Classifier
- ✅ **Deployment**: Gradio App (Interactive UI)
- ✅ **Metric Used**: F1 Score

---

## 📁 Dataset Summary

| Feature                    | Description                                |
|----------------------------|--------------------------------------------|
| `lead_time`                | Days between booking and arrival           |
| `market_segment_type`      | Booking channel (Online/Offline)           |
| `avg_price_per_room`       | Average daily rate                         |
| `no_of_adults`             | Number of adults                           |
| `no_of_weekend_nights`     | Weekend nights booked                      |
| `no_of_week_nights`        | Weekdays booked                            |
| `no_of_special_requests`   | Requests like extra bed, high floor, etc.  |
| `arrival_month`            | Month of arrival                           |
| `required_car_parking_space`| Binary: Needed parking or not             |
| `booking_status` (target)  | Cancelled or Not_Cancelled                 |

---

## 📈 Model Pipeline

1. **Data Preprocessing**:
   - Handled categorical variables using one-hot encoding
   - Removed non-relevant columns like `arrival_date`

2. **Model Building**:
   - Trained a `DecisionTreeClassifier` with stratified train-test split
   - Evaluated with F1 Score

3. **Hyperparameter Tuning**:
   - Tuned `max_leaf_nodes` and `min_samples_split` using `GridSearchCV`
   - Improved generalization from 79.3% → **80.2% F1 Score**

4. **Deployment**:
   - Created a Gradio interface to input booking data and predict cancellations in real-time.

---

## ⚙️ How to Run This Project

### 1. Clone the Repo
git clone https://github.com/dannyj1202/Hotel-Booking-Cancellation-ML-Predictor-Model.git
cd Hotel-Booking-Cancellation-ML-Predictor-Model

### 2. Install Requirements
pip install -r requirements.txt


### 3. Run the Gradio App
python project.py
---

## 🚀 Sample Prediction

| Input               | Value  |
| ------------------- | ------ |
| Lead Time           | 150    |
| Market Segment Type | Online |
| Avg Price Per Room  | 130.50 |
| No. of Adults       | 2      |
| Weekend Nights      | 2      |
| Week Nights         | 3      |
| Special Requests    | 1      |
| Arrival Month       | 7      |
| Required Parking    | Yes    |

→ **Prediction**: ❌ Cancelled
→ **Confidence**: 84.5%

---

## 🎯 Model Performance

| Metric           | Train Data                                        | Test Data |
| ---------------- | ------------------------------------------------- | --------- |
| F1 Score         | 98.96%                                            | **80.2%** |
| Confusion Matrix | ![Confusion Matrix](visuals/confusion_matrix.png) |           |

---

## 📂 Folder Structure

Hotel_Booking_Cancellation/
│
├── visuals/                         # Graphs & screenshots
├── project.py                      # Main Gradio app
├── README.md                       # You’re here!
├── requirements.txt                # Required packages
├── hotel_cancellation_prediction_model_v1_0.joblib
├── model_training.ipynb            # Full data science workflow
├── XYZHotelsGroup.csv              # Dataset

---

## 🧠 Future Improvements

* Integrate model with hotel’s CRM
* Try ensemble models (Random Forest, XGBoost)
* Add support for batch booking predictions

---

## 📜 License

MIT License – use freely, attribute responsibly.

---
OVERALL INDEPTH EXPLANATION: 
Main Problem Statement: 

A chain of hotels, XYZ Hotels Group, is facing a problem of inventory loss due to booking cancellations, resulting in revenue loss. They want your help to build an Data Science solution that will help them predict the likelihood of a booking getting cancelled so that they can take measures to fill in potential vacancies.


• Business Context:

With the increasing popularity and ease-of-access of online hotel booking platforms, customers tend to make reservations in advance to avoid any last minute rush and higher prices. These online platforms offer flexible cancellation options, in some cases even a day before reservation. To compete with this, even offline bookings have increased the flexibility in cancellations. This has led to an increase in the growing number of cancellations, with one of the primary reasons being last-minute changes in travel plans. These sudden changes can result from unforeseen circumstances, such as personal emergencies, flight delays, or unexpected events at the travel destination.

Hotel booking cancellations becomes a crucial problem to solve as it leads to revenue loss and operational inefficiencies. The cancellation of bookings impacts a hotel on various fronts:

1. Loss of revenue when the hotel cannot resell the room

2. Additional costs of distribution channels by increasing commissions or paying for publicity to help sell these rooms

3. Lowering prices last minute, so the hotel can resell a room, resulting in reduced profit margins


• Problem Definition:

A XYZ Hotels Group has been struggling with the challenge of rising cancellations for nearly a year now. However, the last three months they noticed a rise of inventory loss due to cancellation rise to an all-time high of 18%. This has led to a jump in the revenue loss to an all-time high of approx. $0.25 million annually. This has significantly impacted their profit margins.

• In the current context, inventory refers to a hotel room, and the inability to sell one leads to inventory loss

The group has been using heuristic mechanisms (rule and domain expert based) to try and reduce the revenue loss due to cancellations, but this hasn't been effective so far hasn't been effective (neither efficient nor scalable), as evident from the magnitude of losses they are incurring.

The group has decided that they need a Data Science-based solution to predict the likelihood of a booking being cancelled as they expect it to be more effective than their current mechanism. They hope that this proactive approach will help them significantly minimize revenue loss and improve operational efficiency.


Data Gathering:


The data needed for building any Data Science solution is usually obtained from multiple sources.

In the current scenario, we have the following sources:

1. Website Data: This includes information such as website traffic, user interactions, clickstream data, user demographics, and browsing behavior.

2. Property Data: This includes information such as the number of rooms, type of rooms (e.g., single, double, suite), facilities provided (e.g., Wi-Fi, parking, swimming pool), amenities, location details, and any additional property-specific attributes.

3. Agencies Data: This comprises reservation details (e.g., check-in and check-out dates, booking ID), guest profile information (e.g., name, contact details, preferences), cancellation requests, room prices, payment information, and any other relevant data related to the interactions between the property and booking agencies or platforms.

4. Surveys and Feedbacks: This refers to data obtained through customer surveys, feedback forms, and online reviews, which provide insights into customer satisfaction, preferences, and suggestions for improvement.

5. Social Media: This encompasses data collected from various social media platforms such as Facebook, Twitter, Instagram, and LinkedIn, including user posts, comments, likes, shares, and other interactions.

The data from different sources are collected and stored in an organized and secure manner in databases. Databases are made up of tables, which are collections of data organized into rows and columns. The rows represent individual records, and the columns represent the different attributes that make up each record. For example, a table of customer records might have columns for the customer's name, address, phone number, email address, and more.

Once the data is stored in the databases, we can extract necessary data in multiple ways.

1. Export as CSV/Excel File: This method allows for exporting a selected subset or the entire dataset in a CSV (Comma-Separated Values) or Excel file format, which can be easily opened and analyzed using spreadsheet software or using programming languages like Python.

2. Querying from the database: This involves running SQL (Structured Query Language) queries on the database to retrieve specific data based on predefined conditions, allowing for more targeted and customized data extraction for analysis or reporting purposes. The SQL queries can be executed using programming languages like Python by establishing a connection to the database.

Libraries Used :
    I. For reading and manipulating the data:
        a. pandas
        b. numPy
        
    II. For data visualization:
        a. matplotlib
        b. seaborn
        
    III. Making of the ML predictor
        a. sklearn (For splitting the data and building,  tuning and evaluation of model predictor)
        
    IV. Deployment of our Model
        a. os
        b. joblib
        c. gradio
    V. Misc
        a. warnings 

•Further explanation of the role of each library:

1. For Reading and Manipulating Data
    • pandas (pd): Used for working with data tables (like spreadsheets) – read, filter, modify data.
    • numpy (np): Helps with math, especially when working with arrays, numbers, or scientific operations.

2. For Data Visualization (Charts)
    • matplotlib.pyplot (plt): Used for making graphs like bar charts, line charts, histograms, etc.
    • seaborn (sns): Built on top of matplotlib – prettier and easier statistical visualizations.

3. Display Settings for pandas
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
These are formatting tweaks for how data is displayed in your notebook:
    • Show all columns, no matter how many.
    • Limit row display to 200 (instead of 10).
    • Round floating numbers (decimals) to 2 digits (e.g., 65.12345 → 65.12).

4. For Splitting the Dataset
from sklearn.model_selection import train_test_split
    • Splits your dataset into training and test sets.
Example: 80% of the data for training the model, 20% to test how well it works.

5. For Building the ML Model
from sklearn.tree import DecisionTreeClassifier
    • This is the actual ML model you're using: a Decision Tree Classifier.
It’s a simple model that learns yes/no questions to make predictions.

6. For Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
    • This helps find the best settings (called hyperparameters) for your model by trying many combinations automatically.

7. For Evaluating Model Performance
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
These help measure how good your model is:
    • accuracy_score: How many predictions were correct.
    • precision_score: Out of all predicted "canceled" bookings, how many were truly canceled.
    • recall_score: Out of all actual canceled bookings, how many did the model catch.
    • f1_score: A balanced score between precision and recall.
    • confusion_matrix: Shows correct and incorrect predictions in a table.
    • roc_auc_score: Measures how well the model separates the classes (cancel/not cancel).
    • ConfusionMatrixDisplay: A pretty way to show the confusion matrix.
(You import f1_score and make_scorer again, but that’s redundant.)

8. For Saving/Loading the Model & Deployment
import os
import joblib
import gradio as gr
    • os: Lets Python interact with files and folders.
    • joblib: Saves the trained model to a file so you can reuse it later.
    • gradio: A super cool tool that helps you build a web interface to test your model (e.g., sliders, text boxes, etc.)

9. To Hide Warnings
import warnings
warnings.filterwarnings("ignore")
    • This hides annoying warning messages (not errors) to keep your notebook clean.


            
• Exploratory Data Analysis:
    
    Exploratory Data Analysis (EDA) plays a very important role in an end-to-end Data Science solution. It enables
    
    - Understanding the Data: EDA comprehensively reveals the dataset's structure, patterns, and issues. It assesses 
    data quality, identifies missing values, outliers, and inconsistencies, vital for preprocessing and model development.
    
    - Identifying Data Patterns and Insights: EDA discovers meaningful patterns, trends, and relationships through statistical 
    techniques, visualizations, and summarization. It guides model development, hypothesis generation, and decision-making.
    
    - Feature Selection and Engineering: EDA identifies relevant features by analyzing variable relationships, exploring 
    correlations, and visualizing distributions. It improves model prediction by selecting informative features and creating 
    new ones through feature engineering.
    
    Key Feats of our dataset: 

    • The number of adults ranges from 0 to 4, which is usual.
    • The range of the number of weeks and weekend nights seems fine. Though 7 weekends might be a very long stay.
    • At least 75% of the customers do not require car parking space.
    • On average the customers book 85 days in advance. There's also a very huge difference in 75th percentile and maximum value because booking 443 days in advance makes no sense so this indicates that there might be outliers present in this column.
    • The average price per room is 103 euros. There's a huge difference between the 75th percentile and the maximum value which indicates there might be outliers present in this column.
    

• Our data visualization functions: 
    1.  labeled_barplot()
def labeled_barplot(data, feature, perc=False, n=None):
    """
    A function to create a labeled bar chart of a categorical feature.
Parameters:
    - data (DataFrame): The dataset containing the column to plot.
    - feature (str): The name of the categorical column to visualize.
    - perc (bool): If True, show percentage labels; otherwise, show raw counts. Default is False.
    - n (int or None): If specified, only show the top 'n' most frequent categories; otherwise, show all.
Function Logic:
    - Calculates the number of unique values in the given column to adjust plot width dynamically.
    - Uses seaborn's countplot to create a bar chart showing the frequency of each category.
    - For each bar, annotates the top with either:
        a) the count of entries for that category, or
        b) the percentage of the total (if perc=True).
    - Formats axis labels and tick sizes for readability.
    - Useful for quickly understanding the distribution of any categorical feature in the dataset.
Example:
    labeled_barplot(df, 'market_segment_type', perc=True)
    """
 What It Helps You Do:
    • Visually inspect how values in a column are distributed
    • Compare raw counts or relative percentages of each category
    • Limit output to only top n categories (e.g., top 5 room types)

    2. stacked_barplot()
def stacked_barplot(data, predictor, target):
    """
    A function to generate a stacked bar chart that visualizes the distribution of a target variable
    across the different categories of a predictor (independent) variable.
Parameters:
    - data (DataFrame): The dataset containing the predictor and target columns.
    - predictor (str): The categorical feature whose class-wise breakdown you want to analyze.
    - target (str): The classification label column (e.g., booking_status).
Function Logic:
    - Uses pd.crosstab() to compute a contingency table between predictor and target.
        - First, prints a raw count table of occurrences for each (predictor, target) combination.
        - Then, computes a normalized version (row-wise) to get proportions within each predictor class.
    - Uses the normalized table to create a stacked bar chart.
        - Each bar represents a category of the predictor variable.
        - Each bar is split proportionally by the target classes (e.g., Canceled vs Not_Canceled).
    - Labels axes and adjusts tick sizes for readability.
    - Useful for spotting trends in how predictor categories relate to the likelihood of cancellation.
Example:
    stacked_barplot(df, 'market_segment_type', 'booking_status')
    """
What It Helps You Do:
    • Understand the relationship between a feature and the target
    • Visually compare how the proportion of target classes changes across different predictor groups
    • Great for feature exploration in classification tasks

We also noted the following based on the EDA conducted:

- The booking status does not significantly vary with the average price per room
- The monthly booking trends show that the number of bookings rise from January to April, remain consistent from April to July, then rise again till October where it reaches a peak, and then drops down again in November and December

We see that variables like lead time and arrival month have a fair amount of difference for cancelled and non-cancelled bookings, while some of the other variables like market segment and avg price per room do not. But we can only visualize the data in 2 or 3 dimensions, and there may be more complex relationships in the data beyond that which cannot be captured visually.

So, we need an ML model that can do the following:
  - Take the booking detail (customer arrival month, booking lead time, no. of guests, and more) as input
  - Learn the patterns in the input data
  - Fit a mathematical model using these patterns to identify which situations lead to booking cancellation
  - Predict the likelihood of cancellation of a new booking

The ML model is the 'heart' of our Data Science solution.  The model serves as the core component that brings intelligence and functionality to an end-to-end Data Science solution. It leverages learned patterns and insights to generate predictions or perform tasks, enabling organizations to make data-driven decisions, automate processes, and unlock valuable insights from their data.

The model building step of an Data Science solution can be further broken down into the sub-steps shown below.

Data Preprocessing:

Now that we have a better understanding of the data, it's time to preprocess it to ensure it's in the right format for modeling.
Data preprocessing is a crucial step as it enables the following.

- Removing duplicate data: If your data contains duplicate records, it can skew your results. For example, if you are trying to calculate the average sales of a product, and your data includes duplicate sales records, the average will be artificially inflated.

- Correcting errors: If your data contains errors, it can lead to inaccurate results. For example, if your data includes a product with a price of 1000, but the correct price is actually 100, your analysis will be inaccurate.

- Filling in missing values: If your data contains missing values, it can make it difficult to analyze. For example, if you are trying to calculate the average age of a group of people, and your data includes missing ages, you will not be able to calculate an accurate average.

- Transforming the data: Sometimes, it is necessary to transform the data into a different format in order to make it easier to analyze. For example, if your data is in the form of text, you may need to convert it into numbers in order to perform statistical analysis.

For the current scenario, we will be doing the following data preprocessing steps

 1. We will encode the categorical variables using numerical values

   - A computer can understand numbers and not text, so it is important to convert text to numbers

 2. We will also divide the data into two parts - 70% of the data will be used for training purpose and the remaining 30% for testing purpose (This is how we use the concept of validation to fine tune our predictor)

 
Model Training and Evaluation:

During this stage, you gather data and create a model that can make decisions based on that data.
Training an ML model is important because it allows machines to learn and perform tasks without explicit programming. It enables the following:

- Learning from Data: ML models, such as machine learning and deep learning algorithms, learn patterns and make predictions based on data. Through training, the model can identify underlying patterns, correlations, and relationships in the data, enabling it to make accurate predictions or perform specific tasks.

- Generalization and Adaptability: By training an ML model on diverse and representative data, it can learn generalizable patterns and rules that can be applied to new, unseen data. A well-trained model can adapt and make accurate predictions or decisions on new data points it hasn't encountered before. 

- Optimization and Performance Improvement: During the training process, ML models adjust their internal parameters and weights to minimize errors or maximize performance on the training data. Training allows models to fine-tune their internal mechanisms to achieve the best possible performance on the given task.

Note: For a model to generalize well on unseen data we have to ensure that the model is not too complex such that the model is just basically byhearting the data provided and is not even making any decision or a calculated prediction this is also called as overfitting, at the same time we don’t want to make an extremely simple model because that is just as bad as overfitting, also called as underfitting.

We define a confusion matrix function that produces our confusion matrix along with the percentage of total predictions, 

After evaluating the model's performance on the training set we get a  value of 98.96% by using the F1-score
•  F1-score, which is the harmonic mean of:
    ○ Precision: how many predicted positives are truly positive?
    ○ Recall: how many actual positives were correctly predicted?
F1 is a great score when your classes are imbalanced (e.g., way more "Not Canceled" than "Canceled").

So let's further evaluate the model performance by using our remaining 30% test set data entries
We see we get a score of 79.3%

So, what does this tell us
This is a classic sign of overfitting.
Our model performs extremely well on training data but drops significantly on test data — this means it has memorized the training examples but struggles to generalize to new, unseen bookings.

Why It Happens (with Decision Trees)
Decision trees are:
    • Very flexible and powerful
    • But they can easily grow too deep, splitting too much
    • This leads to perfect training accuracy, but poor generalization

So since its overfitting, This becomes a worry as the ultimate goal would be to make predictions for new reservations that have not come in yet, and we do not want a model that will fail to perform well on such unseen data.

We want to minimize these inaccuracies as much as we can without overfitting our predictor at the same time, so this brings us to a key step of AI/ML building which is

• Model Tuning:

Model tuning is important for

- Optimizing Performance: Model tuning allows for finding the optimal configuration of hyperparameters to maximize the performance of the ML model. By systematically adjusting the hyperparameters, such as learning rate, regularization strength, or tree depth, it is possible to find the combination that yields the best results, improving the model's accuracy, precision, recall, or other performance metrics.

- Determining the right fit: Model tuning helps in finding the right set of model parameters that yield the best results. By tuning the model, it is possible to strike a balance and achieve an optimal level of complexity that  ensures that the model neither fails to capture the underlying patterns in the data nor learns the training data too well but fails to generalize to new data.

- Adapting to Data Characteristics:  Model tuning allows for adapting the model to the specific characteristics of the data at hand. Different datasets may require different hyperparameter settings to achieve the best performance. By tuning the model, it becomes possible to adapt to data variations, handle different data distributions, or account for specific data properties, ultimately improving the model's ability to generalize and make accurate predictions.
To improve the model's generalization performance and avoid overfitting, we performed hyperparameter tuning on the DecisionTreeClassifier using GridSearchCV.

• Objective:
Find the optimal combination of tree complexity parameters that balance the model's bias and variance, using F1-score as the evaluation metric due to class imbalance in the target variable (booking_status).

• Parameters Tuned:
We defined a grid of values for two key hyperparameters:
    • max_leaf_nodes: Limits the total number of leaf nodes in the decision tree. Helps control the depth and complexity of the model.
    • min_samples_split: Minimum number of samples required to split an internal node. Higher values prevent the model from overfitting small, noisy splits.

• Evaluation Metric:
We used F1-score as the evaluation metric during tuning, as it gives a balanced measure of precision and recall, which is important in scenarios with potentially imbalanced classes (like Canceled vs Not_Canceled bookings).

• Tuning Method:
We used GridSearchCV with 5-fold cross-validation:
    • Splits the training data into 5 parts
    • Trains and evaluates the model on every possible parameter combination
    • Computes the average F1-score across folds to select the best configuration

• Outcome:
    • The grid search identified the best combination of parameters for our Decision Tree model.
    • The best model was then evaluated on the test data to assess generalization.

•  Benefits of This Tuning:
    • Reduced overfitting compared to the default model
    • Improved the model's performance on unseen (test) data
    • Made the model more stable and generalizable in real-world booking scenarios


• Model Testing: 
Model testing is important for:
- Validating model performance: Testing helps assess how well the model performs under various conditions and scenarios.
- Identifying and mitigating errors or flaws: Testing helps uncover any errors, bugs, or weaknesses in the model.
- Assessing model robustness and generalizability: Testing helps evaluate the model's performance on new, unseen data.
- Building user trust and confidence: Model testing instills trust in the model's capabilities and predictions.

After selecting the best hyperparameters through Grid Search, we evaluated the final tuned Decision Tree model on the unseen test dataset to assess its real-world performance.
• Objective: To measure how well the tuned model generalizes to new, unseen booking data and verify its effectiveness in predicting cancellations.

• Process:
    1. Assigning the Final Model
The best-performing model from the tuning phase (GridSearchCV) is assigned for final evaluation.
    
    2. Generating Predictions
The model makes predictions on the test set using only the input features (X_test).
    
    3. Evaluating Performance
The F1-score was used as the evaluation metric, providing a balanced measure of precision and recall — crucial in this context due to potential class imbalance
    
    4. Results
Model Score on Test Data: 80.2%
        ○ The model achieved an F1-score of 80.2% on the test set.
        ○ This reflects a slight but meaningful improvement over the pre-tuning score of 79.3%, indicating that hyperparameter tuning led to better generalization.
        ○ Compared to the training score of 98.96%, the test score demonstrates reduced overfitting and improved real-world reliability.

Conclusion:
The final model exhibits strong predictive performance on unseen data, balancing the need for accuracy and generalizability. This positions the model as a practical tool for predicting hotel booking cancellations and enabling smarter inventory planning.

• Model Deployment:
There are generally two main modes of making predictions with a deployed ML model:

- Batch Prediction: In batch prediction mode, predictions are made on a batch of input data all at once. This mode is suitable when you have a large set of data that needs predictions in a batch process, such as running predictions on historical data or performing bulk predictions on a scheduled basis.

- Real-time (or Interactive) Prediction: In real-time or interactive prediction mode, predictions are made on individual data points in real-time as they arrive. This mode is suitable when you need immediate or on-demand predictions for new and incoming data.

The choice of prediction mode depends on the specific requirements and use case of the deployed ML model. Batch prediction is preferable when efficiency in processing large volumes of data is important, while real-time prediction is suitable for scenarios that require immediate or interactive responses to new data.

Metrics and dashboarding are the tools that businesses use to track their performance. Metrics are the specific measurements that businesses track. Dashboards are the visual displays of metrics that help businesses to see how they are performing at a glance. Metrics and dashboarding are essential for businesses because they provide the information that businesses need to make better decisions at a glance. By tracking their performance, businesses can identify areas where they are doing well and areas where they need to improve.

Here are some of the benefits of using metrics and dashboarding:
 - Improved decision-making: By tracking their performance, businesses can make better decisions about how to allocate their resources, target their marketing campaigns, and improve their customer service.

- Increased efficiency: By identifying areas where they are doing well and areas where they need to improve, businesses can become more efficient and productive.

- Increased visibility: Dashboards provide a visual display of metrics, which makes it easy for businesses to see how they are performing at a glance.
Improved communication: Dashboards can be used to communicate performance metrics to employees, managers, and stakeholders.

• Decision Making:

Now the final step is to use the ML model for decision-making and determine the impact of implementing the Data Science solution.

The trends of model performance along with the revenue loss incurred is useful for the Data Team. They can use it to: 
- monitor the model's performance over time
- correlate it with financial numbers to gauge the  business impact
- set thresholds for the acceptable lower limit of model performance
- decide when to retrain the model

The property manager can use the dashboard to:
- understand the current status of bookings over a date range
- identify the number of potential vacancies due to likely cancellations
- decide when to stop taking further bookings to avoid overbooking

The leadership can use the dashboard to understand the impact of the Data Science solution in:
- reducing cancellations
- increasing revenue
