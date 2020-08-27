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


 
Conclusion:
From the EDA we can see that diabetes doesn’t have any specific age or BMI level, perhaps all the factors are equally responsible. Patient between the age of 20-30 years has the highest levels of BMI which is over 45 are more prone to diabetes. The pregnant women are more prone to it than normal individuals.  
We can see that using the bagging algorithms, the accuracy of the models increased efficiently. But further performing Boosting Algorithms increased the accuracy to 81% with is the best until now.
