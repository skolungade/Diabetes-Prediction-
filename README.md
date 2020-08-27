# Diabetes-Prediction-
Logistic Regression, Random Forest, Decision Tree, NB, XGBoost

About the Disease:
Diabetes Mellitus often famously called as Diabetes is s a group of metabolic disorders characterized by a high blood sugar level over a prolonged period of time.  It is due to either the pancreas not producing enough insulin, or the cells of the body not responding properly to the insulin produced. The symptoms include increased hunger, increased thirst, weight loss, frequent urination, blurry vision, extreme fatigue, sores that don’t heal. Diabetes is classified as Type 1, Type II and Gestational diabetes.
CDC’s Division of Diabetes Translation report states that 34.2 million Americans—just over 1 in 10—have diabetes and 88 million American adults—approximately 1 in 3—have prediabetes.

Problem Statement:
The dataset chosen by me is for the disease diabetes, we are going to predict where or not a person has diabetes based on the various several medical predictors like insulin level, age, BMI etc. “The objective of the dataset is to predict (diagnostically) whether a patient has diabetes mellitus or not based on certain medical measurements included in the dataset”

About the Dataset:
This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The datasets consist of medical predictor variables such as the number of pregnancies the patient has had, their BMI, insulin level, age, etc. and one target variable, Outcome. 
The following are the attributes and their information:
1.	Number of times pregnant
2.	Plasma glucose concentration 2 hours in an oral glucose tolerance test
3.	Diastolic blood pressure 
4.	Triceps skin fold thickness
5.	Insulin levels
6.	Body mass index 
7.	Diabetes pedigree function
8.	Age in years
9.	Outcome variable (0 or 1)
Its observed that there are 768 observations with 9 variables

Steps performed:
1.	Loading and understanding the data
2.	Exploratory Data Analysis
3.	Cleaning the data
4.	Training and Testing 
5.	Combine Model Predictions into Ensemble Predictions
6.	Feature Selection and XGBoost

1.	Loading and understanding the data:
I used python to load the csv file and tried to understand it.
Its observed that there are 768 observations with 9 variables
Further, I found out the 5-point summary stats 

2.	Exploratory Data Analysis
All our variables are positively correlated with different degree of correlation. Glucose has the highest correlation with Outcome and BloodPressure has the lowest
Body Mass Index is a calculation which has two components, height and weight of a person. Patient between the age of 20-30 years has the highest levels of BMI which is over 45. High BMI patients tends to have high chances of Outcome 1.
-> As we can see from our distribution plots, Glucose, BloodPressure, SkinThickness, and Insulin all have 0 values. Technically the values of BP, Glucose and Insulin cannot be zero which means that the data is incorrect we will further deal with it which preprocessing/cleaning the data.
-> Pair Plot can be used to visualize and analyze one variable with respect to another. The diagonal is a histogram which shows distribution of a single variable. We can see that there are many outliers with our data. Here hue is taken as Outcome, which clearly distinct the Outcome 0 and Outcome 1 in our visualization.

3.	Data Cleaning
As mentioned above, there are several variables which have 0 values, that is the values are 0 which are not possible. I performed the following steps to deal with the problem
1.	Replaced the 0 with Nan. This is done because when the mean of the variable is calculated, the data field with 0 value will not be considered. Doing this will give the true mean of the valid data entry.
2.	In the next step, we replace the value of NaN with our calculated mean.(the ideal method to deal with missing values is either enter mean, media or mode, we use mean)
After doing this we plot the distribution plot and observe that in glucose, BMI, BP the values aren’t 0 and it has been fixed.
 
4.	Training and Testing Models
Previously, I have performed Data Cleaning and EDA for the diabetes dataset. This week I will perform some modelling on that data and get the accuracy of each of the model. This week I will be performing some of the supervised machine learning algorithms on our dataset. 
The few models I will be performing are:
1.	Logistic Regression
2.	Decision tree Classifier
3.	Naive Bayes
4.	Gradient Boost
5.	Random Forest
To begin any modeling the approach is to Split the data into training and testing dataset. To split the data in Python, I will be using pandas, train_test_split() function. I split it by 70:30 ratio and further I trained the models using the dataset X_train and Y_train and then tested the other two datasets, X_test and Y_test. We usually fit the model on the training data and get predictions and accuracy from the test data.
What is a Supervised Learning Algorithm?
Supervised learning is the most common subbranch of machine learning. They are designed to learn by example.
While we train a supervised learning algorithm, the training data will consist of inputs paired with the correct outputs. During the training process, the algorithm will look for patterns in the data that link with the anticipated outputs. 
After the training process, any supervised learning algorithm will take new unseen inputs and will determine which label these new inputs will be classified as based on the earlier training dataset. The main aim of a supervised learning algorithm is to predict the correct label for new input dataset. 
A supervised learning algorithm can be written simply as:
Y=f(x)
Supervised learning can be split into two subcategories: Classification and Regression.
The accuracy scores of the Supervised Algorithms on the diabetes dataset is as follows:
 
We can see that Random Forest has a highest Accuracy of 0.745 ~ 0.74 or 74% and Decision Tree Classifier has the least Accuracy of 0.68 or 68%
The below box plot visualization shows the 5-point summary statistics of each of the following Models. We can see that Random Forest has the best Summary stats.
 
5.	Combine Model Predictions into Ensemble Predictions
From the above results we saw that Random Forest and Decision Tree Classifier has the highest accuracy. We will further move ahead with three most popular methods for combining the predictions from different models which is Bagging, (building multiple models, characteristically of the same type from different subsamples of the training dataset.)
The final output prediction is averaged across the predictions of all the sub-models.
The three bagging models that I have performed are as follows:
1.	Bagged Decision Trees
2.	Random Forest Classifier

1.	Bagged Decision Tree
A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.
Here we can observe that we achieved an accuracy score of 0.74 ~ 74%
Using Bagged Algorithm for the decision trees increased its accuracy from 68% to 74%
2.	Random Forest Classifier
A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
Here we can observe that we achieved an accuracy score of 0.7575 ~ 76%
Using bagged algorithm for Random Forest Classifier the accuracy increased from 64% to 76%

6.	Feature Selection and XGBoost
Earlier we did not select any features, now we will go ahead and perform feature selections and then perform Gradient Boost to increase the accuracy.

Feature Selection:
Feature selection is also called variable selection or attribute selection. It is the automatic selection of attributes in your data (such as columns in tabular data) that are most relevant to the predictive modeling problem you are working on.
Feature selection methods aid you in your mission to create an accurate predictive model. They help you by choosing features that will give you as good or better accuracy whilst requiring less data.
Feature selection methods can be used to identify and remove unneeded, irrelevant and redundant attributes from data that do not contribute to the accuracy of a predictive model or may in fact decrease the accuracy of the model.
Correlation
If we fit highly correlated data in our model, it results in the overfitting problem. Thus, for example if there are two highly correlated features, we must drop the one that has more correlation with other feature.
There is no any highly correlated feature in this data set.

We will further check with variables are important based on Bagged decision trees like Random 
Forest and Extra Trees. We can see that Glucose is the most important variable.

 
Now we will move with Algorithm Tuning and XG Boost, for improving the prediction
XGBoost:
XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. In prediction problems involving unstructured data (images, text, etc.) artificial neural networks tend to outperform all other algorithms or frameworks. 
 
Using XG Boost we got an accuracy of 81% with is far most the best we could reach up to.
 
Conclusion:
From the EDA we can see that diabetes doesn’t have any specific age or BMI level, perhaps all the factors are equally responsible. Patient between the age of 20-30 years has the highest levels of BMI which is over 45 are more prone to diabetes. The pregnant women are more prone to it than normal individuals.  
We can see that using the bagging algorithms, the accuracy of the models increased efficiently. But further performing Boosting Algorithms increased the accuracy to 81% with is the best until now.
