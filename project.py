# libraries to help with reading and manipulating data
import pandas as pd
import numpy as np

# libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# removing the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
# setting the limit for the number of displayed rows
pd.set_option("display.max_rows", 200)
# setting the precision of floating numbers to 2 decimal points
pd.set_option("display.float_format", lambda x: "%.2f" % x)

# library to split data
from sklearn.model_selection import train_test_split

# library to build ML model
from sklearn.tree import DecisionTreeClassifier

# library to tune ML model
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

# libraries to evaluate the ML model
from sklearn.metrics import f1_score, make_scorer

# libraries to deploy the ML model
import os
import joblib
import gradio as gr

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# # loading the data into a Pandas dataframe
hotel = pd.read_csv("Hotel_Booking_Cancellation_Model_Predictor/XYZHotelsGroup.csv")
#print(hotel.sample(10 , random_state=10)) // displaying 10 random rows from the dataset

#creating a copy of the dataframe to work with to avoid changing the original dataframe
data = hotel.copy()
#print(data.describe().T) // displaying the summary statistics of the dataset

# defining a function to create a bar graph with percentage values
def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe (our dataset)
    feature: dataframe column (the feature aka column to plot)
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n (most frequent) category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # returns length of the column
    count = data[feature].nunique() #returns the number of unique values in the column

    #adjusts the figure width dynamically based on how many categories you’re plotting.
    # if n is None, it will plot all categories, otherwise it will plot the top n categories
    if n is None:
        plt.figure(figsize=(count + 2, 6))
    else:
        plt.figure(figsize=(n + 2, 6))

    # plotting the bar chart
    # sns.countplot() is used to plot the count of each category in the column
    # data: dataframe to plot
    # x: column to plot
    # palette: color palette to use for the bars
    # order: order of the categories to plot, here we are plotting the top n categories
    # if n is None, it will plot all categories, otherwise it will plot the top n categories
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n],
    )
    

    for p in ax.patches:    
        """This loop:
        Goes through each bar (patch) in the chart
        
        Calculates:
            • Percentage = (height of bar / total count) * 100
            • Or just raw count = height of bar

        Adds a label on top of each bar with ax.annotate(...)"""
    # looping through each bar in the chart
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=15,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    #Formatting the Plot
    # increase the size of x-axis and y-axis scales
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)

    # setting axis labels
    ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=15)
    ax.set_ylabel('')

    # show the plot
    plt.show()


"""the stacked_barplot function generates a stacked bar chart to show how each category of the predictor variable is split across the target classes 
(like how many bookings were canceled vs not canceled for each market_segment_type, etc.)"""
def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique() # returns the number of unique values in the predictor column
    sorter = data[target].value_counts().index[-1] # gets the last value in the target column, which is used to sort the categories in the plot
    tab1 = pd.crosstab(data[predictor], data[target], margins=True) # creates a crosstab of the predictor and target columns, with margins (totals) included
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index") # creates a crosstab of the predictor and target columns, with percentages normalized by index (i.e., each row sums to 1)
    tab.plot(kind="bar", stacked=True, figsize=(count + 5, 5)) # plotting a stacked bar chart of the crosstab
    plt.legend(loc="upper left", fontsize=12, bbox_to_anchor=(1, 1)) #It generates a stacked bar chart to show how each category of the predictor variable is split across the target classes (like how many bookings were canceled vs not canceled for each market_segment_type, etc.)

    # setting the formatting for x-axis
    plt.xticks(fontsize=15, rotation=0)
    plt.xlabel(predictor.replace('_', ' ').title(), fontsize=15)
    # setting the formatting for y-axis
    plt.yticks(fontsize=15)
    # show the plot
    plt.show()


# visualizing the number of cancelled bookings
#labeled_barplot(data, "booking_status", perc=True)

# visualizing the relationship between lead time and booking cancellation
plt.figure(figsize=(8, 5))
sns.boxplot(data=data, x="booking_status", y="lead_time")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('')
plt.ylabel('Lead Time', fontsize=15)
#plt.show()

# visualizing the relationship between avg room price and booking cancellation
plt.figure(figsize=(8, 5))
sns.boxplot(data=data, x="booking_status", y="avg_price_per_room")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('')
plt.ylabel('Avg Price per Room', fontsize=15)
#plt.show()

#stacked_barplot(data, "market_segment_type", "booking_status")

# converting the 'arrival_date' column to datetime type
data['arrival_date'] = pd.to_datetime(data['arrival_date'])

# extracting month from 'arrival_date'
data['arrival_month'] = data['arrival_date'].dt.month

# grouping the data on arrival months and extracting the count of bookings
monthly_data = data.groupby(["arrival_month"])["booking_status"].count().to_frame().reset_index() 
#The above line groups the data by 'arrival_month' and counts the number of bookings for each month, then resets the index to convert the result into a DataFrame.
# renaming the columns for better readability
monthly_data.columns = ['Month', 'Bookings']
#print(monthly_data)

# visualizing the trend of number of bookings across months
plt.figure(figsize=(10, 5))
sns.lineplot(data=monthly_data, x="Month", y="Bookings")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('')
plt.ylabel('# Bookings', fontsize=15)
#plt.show()

##Let's check the percentage of bookings canceled in each month
#stacked_barplot(data, "arrival_month", "booking_status")

"""We also noted the following based on the EDA conducted:

- The booking status does not significantly vary with the average price per room
- The monthly booking trends show that the number of bookings rise from January to April, remain consistent from April to July, then 
rise again till October where it reaches a peak, and then drops down again in November and December

We see that variables like lead time and arrival month have a fair amount of difference for cancelled and non-cancelled bookings, while some
 of the other variables like market segment and avg price per room do not. But we can only visualize the data in 2 or 3 dimensions, and there 
 may be more complex relationships in the data beyond that which cannot be captured visually.

So, we need an ML model that can do the following:
  - Take the booking detail (customer arrival month, booking lead time, no. of guests, and more) as input
  - Learn the patterns in the input data
  - Fit a mathematical model using these patterns to identify which situations lead to booking cancellation
  - Predict the likelihood of cancellation of a new booking
"""
# encoding the output (also called target) attribute
data["booking_status"] = data["booking_status"].apply(
    lambda x: 1 if x == "Canceled" else 0
) # # converting the booking status to binary values (1 for cancelled, 0 for not cancelled) this concept is known as classification in supervised learning
# separating the input and output variables
X = data.drop(["booking_status","arrival_date"], axis=1) #  dropping the booking status and arrival date columns from the input data
y = data["booking_status"] # # extracting the booking status column as the output variable
#notice how we have an input and output class label, which is a requirement for supervised learning

# encoding the categorical variables
X = pd.get_dummies(X, drop_first=True) #This converts categorical columns (like market_segment_type) into numeric dummy variables using One-Hot Encoding.

# splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
"""This splits your dataset into:
X_train / y_train: 70% of data for training the model
X_test / y_test: 30% of data for testing/evaluating the model

random_state=42: ensures that every time you run this, the split is exactly the same (reproducible)
"""

#print(X.head())

# defining the ML model to build
model = DecisionTreeClassifier(random_state=1)  
"""This line creates an instance of a Decision Tree Classifier, which is a popular algorithm for 
classification problems (like predicting Canceled vs Not_Canceled).
The model is not trained yet — this is just declaring the model."""

# So let's training the ML model on the train data
model.fit(X_train, y_train)

def confusion_matrix_sklearn(model, predictors, target):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    """
    y_pred = model.predict(predictors)
    #The model uses the predictor data (e.g., X_test) to make predictions (y_pred) — this is your model output.
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)
    """cm.flatten() turns the 2x2 matrix into a flat list: [TN, FP, FN, TP]
       For each value (item), it creates a label string with:
        • The count (e.g., "123")
        • A newline
        • The percentage of total predictions (e.g., "35.4%")
       reshape(2, 2) turns the label list back into a 2x2 format to match the confusion matrix
"""
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

# confusion matrix for train data
#confusion_matrix_sklearn(model, X_train, y_train)

# evaluating the model performance on the train data
model_train_predictions = model.predict(X_train)
model_train_score = f1_score(y_train, model_train_predictions)

#print("Model Score on Train Data:", np.round(100*model_train_score, 2))
#We get a value of 98.96%

# confusion matrix for test data
#confusion_matrix_sklearn(model, X_test, y_test)

# evaluating the model performance on the test data
model_test_predictions = model.predict(X_test)
model_test_score = f1_score(y_test, model_test_predictions)

#print("Model Score on Test Data:", np.round(100*model_test_score, 2))
#We now get a value of 79.3%

# choosing the type of ML Model
dummy_model = DecisionTreeClassifier(random_state=1, class_weight="balanced")
"""This creates a base Decision Tree model, random_state=1 ensures reproducibility.
class_weight="balanced" automatically adjusts weights of classes based on their frequencies:
Useful if your dataset is imbalanced 
"""

# defining the grid of parameters of the ML Model to choose from
parameters = {
    "max_leaf_nodes": [150, 250],
    "min_samples_split": [10, 30],
}
"""These are the hyperparameters we're going to test:
• max_leaf_nodes: Limits how many leaf nodes (final decision points) the tree can have.
    Prevents overfitting by stopping tree growth early.
• min_samples_split: Minimum number of samples needed to split an internal node.
    Hgher values = more conservative splits = less overfitting.

    Grid Search will try every combination of these values:
        • (150, 10)
        • (150, 30)
        • (250, 10)
        • (250, 30)
"""

# defining the model score on which we want to compare parameter combinations
acc_scorer = make_scorer(f1_score) #Here we are going to evaluate all our parameter combinations by using an F1-measure which is typically ideal for classification especially with imbalanced classes

# running the model tuning algorithm
grid_obj = GridSearchCV(dummy_model, parameters, scoring=acc_scorer, cv=5)
grid_obj = grid_obj.fit(X_train, y_train)
"""Now GridSearchCV takes your model and hyperparameter combinations
    For each combination:
        • It trains and tests the model 5 times (because cv=5)
        • Each time, it splits the training set differently to simulate real-world variance
    
    At the end, it selects the best parameter set based on average F1 score across the folds
    
    grid_obj now holds the best version of our model trained with the best parameters."""

# selecting the best combination of parameters for the model to create a new model
tuned_model = grid_obj.best_estimator_

# training the new ML Model
tuned_model.fit(X_train, y_train)

# evaluating the model performance on the train data
tuned_model_train_predictions = tuned_model.predict(X_train)
tuned_model_train_score = f1_score(y_train, tuned_model_train_predictions)

#print("Model Score on Train Data:", np.round(100*tuned_model_train_score, 2))
#Model Score on Train Data: 81.83

# evaluating the model performance on the test data
tuned_model_test_predictions = tuned_model.predict(X_test)
tuned_model_test_score = f1_score(y_test, tuned_model_test_predictions)

#print("Model Score on Test Data:", np.round(100*tuned_model_test_score, 2))
#Model Score on Test Data: 80.2

#NOW the training set and test set performances are much more similar now, so we can say that the model is able to generalize well

final_model = tuned_model

# evaluating the model performance on the test data
final_model_test_predictions = final_model.predict(X_test)
final_model_test_score = f1_score(y_test, final_model_test_predictions)

"""Now that the model is finalized, you use it to make predictions on the test dataset (X_test).
This simulates how the model would behave in the real world — predicting whether future hotel bookings will be canceled."""
#print("Model Score on Test Data:", np.round(100*final_model_test_score, 2))
#Model Score on Test Data: 80.2

"""What This Tells Us?
    The test F1 score increased after hyperparameter tuning, from 79.3% → 80.2%. That's a solid win!
    The training score remains much higher, but the gap is now slightly reduced, which suggests:
        Our tuned model generalizes a bit better
        Overfitting has been reduced, though it could still be improved further

     Why This Is Good?
        Even a 1% increase in F1-score on test data can be meaningful in real-world ML tasks, especially if 
        it leads to fewer wrong predictions about cancellations.
"""

# exporting the final model to the disk
joblib.dump(final_model, 'hotel_cancellation_prediction_model_v1_0.joblib')

# loading the final model from the disk
cancellation_predictor = joblib.load('hotel_cancellation_prediction_model_v1_0.joblib')

# define a function that will take the necessary inputs and make predictions
def predict_cancellation(lead_time, market_segment_type, avg_price_per_room, no_of_adults, no_of_weekend_nights, no_of_week_nights, no_of_special_requests, arrival_month, required_car_parking_space):

    # dictionary of inputs
    input_data = {
        'lead_time': lead_time,
        'no_of_special_requests': no_of_special_requests,
        'avg_price_per_room': avg_price_per_room,
        'no_of_adults': no_of_adults,
        'no_of_weekend_nights': no_of_weekend_nights,
        'required_car_parking_space': 1.0 if required_car_parking_space == "Yes" else 0.0,
        'no_of_week_nights': no_of_week_nights,
        'arrival_month': arrival_month,
        'market_segment_type_Online': 1 if market_segment_type == 'Online' else 0,
    }

    # create a dataframe using the dictionary of inputs
    data_point = pd.DataFrame([input_data])

    # predicting the output and probability of the output
    prediction = cancellation_predictor.predict(data_point).tolist()
    """cancellation_predictor: This is your trained and loaded model.
       .predict(data_point): Makes a prediction for the single input row (e.g., "Canceled" or "Not_Canceled").
        tolist(): Converts the prediction result (usually a NumPy array like [1]) into a plain Python list → e.g., [1] or [0].
       
       This prediction list holds a single element:
        1 = booking is canceled
        0 = booking is not canceled
"""
    prediction_prob = np.round(100*cancellation_predictor.predict_proba(data_point)[0][0], 2) if prediction == 1 else np.round(100*cancellation_predictor.predict_proba(data_point)[0][1], 2)
    """This line does two things:   
        Calls .predict_proba(data_point) to get the probability scores for both classes.
        Based on the predicted class (0 or 1), it picks the correct probability.

        Let’s break it further:
        predict_proba(...) returns something like:  [[0.27, 0.73]]  # [probability of Not_Canceled, probability of Canceled]
        Now Based on the prediction:
            • If prediction == 1 (i.e., model predicted Canceled), then we want the probability for class 1 → 0.73
            • If prediction == 0, we want the probability for class 0 → 0.27
        
            And lastly we just convert it to a percentage
"""
    # returning the final output
    return ("Yes", str(prediction_prob)+"%") if prediction[0] == 1 else ("No", str(prediction_prob)+"%")

"""This function performs the following:
• Accepts user inputs like: lead_time, market_segment_type, avg_price_per_room, etc.
• Encodes categorical variables (Yes → 1, No → 0, Online → 1, etc.)
• Packs inputs into a DataFrame (to match training format)

• Calls the model to:
    • Predict whether the booking will be cancelled (Yes/No)
    • Calculate the probability/confidence of that prediction

• Returns both outputs as strings (e.g., "Yes", "84.2%")"""


# creating the deployment input interface, All of this comes with gradio library and was used to help me get a proper interactive UI to further test my predictor's performance
model_inputs = [
    gr.Number(label="Lead Time"),
    gr.Dropdown(label="Market Segment Type", choices=["Online", "Offline"]),
    gr.Number(label="Average Price per Room"),
    gr.Number(label="Number of Adults"),
    gr.Number(label="Number of Weekend Nights"),
    gr.Number(label="Number of Week Nights"),
    gr.Number(label="Number of Special Requests"),
    gr.Dropdown(label="Arrival Month", choices=np.arange(1,13,1).tolist()),
    gr.Dropdown(label="Required Car Parking Space", choices=["Yes", "No"])
]

# creating the deployment output interface
model_outputs = [
    gr.Textbox(label="Will the booking be cancelled?"),
    gr.Textbox(label="Chances of Cancellation")
]

# defining the structure of the deployment interface and how the components will interact
demo = gr.Interface(
    fn = predict_cancellation,
    inputs = model_inputs,
    outputs = model_outputs,
    allow_flagging='never',
    title = "Hotel Booking Cancellation Predictor",
    description = "This interface will predict whether a given hotel booking is likely to be cancelled based on the details of the booking.",
)

# deploying the model
demo.launch(inline=False, share=True, debug=True)

# shutting down the deployed model
demo.close()