# Investigation of Water Potability

## Project Motivation

According to the *Global Burden of Disease* study 1.2 million people died prematurely as a result of unsafe drinking water. This number is more than three times the amount of homicides globally and equivalent to the total amount of road deaths (1). 

![Image](Images/death-tolls.png)

It's clear that unsafe drinking water is a problem for global health. In this project, I will be using Decision Trees and Random Forests to classify water as either potable or unpotable. I will also be attempting to analyze which features are most important in predicting water potability. 


## Predicting Water Potability with Decision Trees and Random Forests

This data set was found on Kaggle.
https://www.kaggle.com/datasets/adityakadiwal/water-potability

After reading in the data and importing the necessary libraries, this is what our data looks like. 


![Image](Images/water-df.png)


The data consists of 3,276 water samples and contains 9 numeric values. 

    1. pH Value 
    2. Hardness
    3. Solids
    4. Chloramines
    5. Sulfate
    6. Conductivity
    7. Organic_carbon
    8. Trihalomethanes
    9. Turbidity
    10. Potability

**What are the features?**
1. **PH Value** represents the acid-base balance of the water. WHO recommends a permissible PH range of 6.5 to 8.5. All data points in this data set are between 6.52 and 6.83 which are all within the range of the WHO. 

2. **Hardness** is caused by calcium and magnesium deposits seeping into water from geologic deposits. 

3. **Solids** represents the total dissolved solids found in the water which can include disolved organic materials and inorganic materials.

4. **Chloramines** represents total chlorine levels found in water. Chlorine and chloramine are the major disinfectants used in public water systems. Chloramines are most commonly formed when ammonia is added to chlorine to treat drinking water. Chlorine levels up to 4 milligrams per liter (mg/L or 4 parts per million (ppm)) are considered safe in drinking water.

5. **Sulfate** are naturally ocurring substances found in minerals, soil and rocks. Sulfate concentration in freshwater is about 3 to 30 mg/L in freshwater supplies, yet in saltwater supplies the range is about 2,700 mg/L. 

6. **Conductivity** measures the ionic process of a solution that it enables it to transmit current. According to WHO standards, EC value should not exceeded 400 μS/cm.

7. **Organic_carbon** is a measure of the total amount of carbon in organic compounds in pure water. According to US EPA < 2 mg/L as TOC in treated / drinking water, and < 4 mg/Lit in source water which is use for treatment.

8. **Trihalomethanes** are chemicals which may be found in water treated with chlorine. The concentration of THMs in drinking water varies according to the level of organic material in the water, the amount of chlorine required to treat the water, and the temperature of the water that is being treated. THM levels up to 80 ppm is considered safe in drinking water.

9. **Turbidity** is a measure of light emitting properties of water and the test is used to indicate the quality of waste discharge with respect to colloidal matter. The mean turbidity value obtained for Wondo Genet Campus (0.98 NTU) is lower than the WHO recommended value of 5.00 NTU.

10. **Potability** value of 1 indicates Potable (safe for human consumption) and 0 indicates not potable (not safe for human consumption).


### Checking class balance in our data set

![Image](Images/percent-bar-graph.png)

In this case, our data set is slightly imbalanced with class 0 representing 61.0% of our population and class 1 representing 39.0% of our data. Because the data is not heavily imbalanced we will simply proceed for now, but later we can attempt to alter our data set in order to improve model performance. 

#### Dealing with Class Imbalance
1. **Oversampling.** The goal of this technique is to increase increase the represenation of the minority class by generating synthetic data. There are several methods by which we can generate synthetic data. 

2. **Undersampling.** This process involves randomly getting rid of data points of the majority class until the two classes are balanced.

3. **Class weighting.** This process involves assigning higher weights to the minority class during model training.

4. **SMOTE (Synthetic Minority Over-sampling Technique)** This process generates synthetic data of the minority class by identifying minority data points and neighboring data points and creating synthetic points via some form of interpolation. 

#### Creating a balanced data set

```
# create a balanced data set 

import pandas as pd
from sklearn.utils import resample

# Assuming your original dataset is stored in a DataFrame called 'df'
# Separate majority and minority classes
majority_class = df[df['Potability'] == 0]  # Replace 'target' with the name of your target column
minority_class = df[df['Potability'] == 1]


# Undersample majority class
undersampled_majority = resample(majority_class,
                                 replace=False,  # Set to False to perform undersampling without replacement
                                 n_samples=len(minority_class),  # Set the number of samples to match the minority class
                                 random_state=42)  # Set a random state for reproducibility

# Combine undersampled majority class with the minority class
balanced_df = pd.concat([undersampled_majority, minority_class])

# Shuffle the dataset to randomize the order
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

```

![Image](Images/balanced-bar-graph.png)


Checking for a correlation between different variables. 

![Image](Images/pairplot.png)

There doesn't seem to be a clear correlation between any of the variables.

### Preparing our Data

Let's check for missing data. 

![Image](Images/missing-values.png)

All the variables are numerical so we don't need to convert any of the data; however, there is quite a bit of missing data so we need to deal with those values. 


### Dealing with the missing values
 
As we can see there are a good amount of missing values that we need to deal with. Each variable has different amounts of missing values so the values appear to be missing at random. Further, some of our variables appear to be missing signficant shares of their values (3276-2785/3276 ph) and (3276-2495/3276 Sulfate) so it isn't appropriate to simply delete the rows as we would lose a great amount of our training data. 

Based off our high amount of missing values and the randomness of the values, mean imputation, median imputation, and most frequency imputation seem most appropriate. We can use each imputation method and later assess model accuracy on each set of data. 

```
df_mean = df.apply(lambda x: x.fillna(x.mean()))
df_median = df.apply(lambda x: x.fillna(x.median()))
df_mode = df.apply(lambda x: x.fillna(x.mode()))
```

After replacing all the NaN values with the respective Mean and Median of the feature the data contained no missing values as is seen below. 
![Image](Images/mean-imputation.png)

My guess is that the median value will be most accurate because it better accounts for extreme values, and mode imputation tends to be better for categorical data.

### Problem with Mode Imputation



 I ran into a problem when attempting to replace all of the values with the mode (most frequent value). In many cases, there can be multiple "most frequent values", and thus there are several instances in which the mode method returns a series and not a singular value. Thus, there are still many NaN values. To avoid simply dropping these values we will not be using mode imputation and instead only rely on median and mean imputation.

![Image](Images/mode-imputation.png)

In the future, we can also experiment with more advanced types of imputation methods such as **K Nearest Neighbors Imputation** or **Regression Imputation**. 

## Implementing and Evaluating Machine Learning Models

### Train Test Split

```
from sklearn.model_selection import train_test_split
X = df_mean.drop('Potability',axis=1)
y = df_mean['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
```

Here we are splitting the data into a "training" set, data which we will train our machine learning algorithm on, and a "testing" set, data which we will hold back in order to see the how accurate our model performed. 

### Decision Tree Model Theory

Decision trees can be used for both regression and classification tasks and is a form of supervised learning because it requires the data to be labeled in order to create a mode. In the case of classfication, the logical statement at each step of the decision tree is decided on which condition will maximize information gain (such as entropy or gini impurity?). This process continues recursively until all data is classified into its respective class. 

In the case of regression, we are attempting to predict a numeric value rather than classify data. Here the algorithm selects which feature will optimize variance reduction such as mean squared reduction or mean absolute error (essentially a loss function). (?) (Optimizes for loss fucntion rather than information gain?). 

**Problems with decision trees**
Decision trees are prone to overfitting. Overfitting is when a machine learning model learns from a set of training data "too well" and consequently is unable to replicate well on unseen data. Essentially, the model becomes so accurate on the training data that it is highly biased when assessing new data. Decision trees are especially prone to overfitting when there are many features in our data or if the tree grows too deep. The deeper the tree grows, the more likely overfitting will occur. Other problems that can lead to overfitting in decision trees arise from the size and distribution of our data. Overfitting can occur when our data is too small or when it's unbalanced. 

Techniques to avoid overfitting 
- pruning
- maximum tree depth
- random forests

**Random Forest Model**
In order to address the common problem of overfitting that is common with decision trees we can use a machine learning method known as random forests. 

A random forest model essentially works by creating a series of different decision trees and taking the majority decision of the trees. The process of implementing a random forest is as follows.
1. Randomly sleect a subset of the training data (with replacement) to create a bootstrapped sample. 
- In this case, "with replacement" refers to the fact that data can be resampled in multiple constructed decision trees. For example, if you have a sample of 200 data points and select a subset of 100 data points for each tree with replacement, each data point has a chance of being selected more than once. Further, it is possible that some data points will be selected multiple times while others not at all. The selection of this data is done completely at randomly and every data sample from the original has an equal chance of being selected for each new dataset. 

2. Randomly select a subset of features to consider at each split of the decision tree. How many features can be considered (?) 
3. Create a decision tree based on each bootstrapped set of data using the selected features. 

4. REPEAT PROCESS

5. To make a prediction, pass a data point into all the decision trees and take the majority vote from the trees. 

*Interesting that the term bootstrapping refers to "pull oneself up by the bootstraps" or use one's resources in order to achieve a goal. In statistics the term bootstrapping is used to refer to the idea of using the data itself to estimate its properties without assuming the underlying distribution.*


How many trees are used in a random forest? 
- it depends and there is a point of diminishing returns in which adding new trees doesn't necessarily improve the generalization performance of the model. 
- common practices
    - grid search? 
    - randomized search? 

- number of trees is a hyperparameter

![Image](Images/decision-tree.png)


### Decision Tree Model Implementation
```
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
```

#### Predict and Evaluate Decision Tree

```
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
```

#### Interpreting Classification Reports

##### df_mean
![Image](Images/classification1.png)

##### df_median
![Image](Images/classification2.png)


**Classification report values**

Precision: Precision measures what percent of predicted true values of a particular class are indeed true. In this case for class 0, the precision is 0.68 which means that 68% of the instances the model predicted to be as class 0 (water not potable) are indeed class class 0 instances. 

Recall: (True Postitive Rate) 



**Recall** also known as the sensitivity or true positive rate, calculates the percentage of true positive instances which are correctly by the model. In the case of Random Forest (median imputation), the model correctly identified 89% of actual instances belonging to class 0, while only recalling 34% of of actual instances of class 1. *(Why such a large disparity)*

**F-1 Score:** F-1-score is the harmonic mean of precision and recall metrics. Harmonic means are useful for addressing imbalance and give more value to smaller values, thus reducing the impact of outliers. In this case we want to balance precision and recall, and thus, use the harmonic mean. 

**Support:** Support metric is simply the number of instances we have in our data set of each class. 

**Accuracy:** Accuracy calculates the model's predictions regardless of the class. Thus, an accuracy value of 0.68 means that 68% of all instances of all instances were correctly classified. 

**Macro Avg:** Macro Avg calculates the average precision, recall and F-1 score across classes. 

**Weighted Avg:** Weighted Avg calculates the weighted average of precision, recall and F-1 considering the support (number of instances of each class). 

```
print(confusion_matrix(y_test,predictions))

[[542  61]
 [243 137]]
 ```

**Precision**

Class 0: 542 / (542 + 243) = 0.69
Actual 0's / Total Predicted 0's

Class 1: 137 / (137 + 69) = 
Actual 1's / Total Predicted 1's
*precision refers to how accuraate the models predictions were*

**Recall**

Class 0: 542 / 603 = 0.90
Correctly predicted 0's / Total 0's

Class 1: 137/380 = 0.36
Correctly predicted 1's / Total 1's
*recall refers to what percent of the values were correctly predicted*

**F-1 Score (Harmonic Mean)**
?



              precision    recall  f1-score   support

           0       0.69      0.90      0.78       603
           1       0.69      0.36      0.47       380

    accuracy                           0.69       983
    macro avg      0.69      0.63      0.63       983
    weighted avg   0.69      0.69      0.66       983

### Data: Source? 

### Works Cited
1. https://ourworldindata.org/water-access


