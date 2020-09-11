ID3 Decision Tree Algorithm 
===================


ID3 is a Machine Learning Decision Tree Algorithm that uses two methods to build the model. The two methods are Information Gain and Gini Index.
This Model consists of only Information Gain. Next update will include Gini Index. 

----------


Installation
-------------
Install directly from my [PyPi](https://pypi.org/project/classic-ID3-DecisionTree/)

> pip install classic-ID3-DecisionTree

Or Clone the [Repository](https://github.com/safir72347/ML-ID3-Decision-Tree-Classification-Library-PyPi) and install

> python3 setup.py install

Parameters
-------------

## * X_train 
-------------
The Training Set array consisting of Features.

## * y_train
-------------
The Training Set array consisting of Outcome.

## * dataset
-------------
The Entire DataSet.


Attributes
-------------

## * information_gain(X_train, y_train)
-------------
Initialise the Information Gain class with training set.

## * add_features(dataset, result_col_name)
-------------
Add the features to the model by sending the dataset. The model will fetch the column features. The second parameter is the column name of outcome array.

## * decision_tree()
-------------
To build the decision tree

## * predict(y_test)
-------------
Predict the Test Set Results


<i class="icon-file"></i> Documentation
-------------

### 1.  Install the package
>  pip install classic-ID3-DecisionTree

### 2. Import the library
>  from classic_ID3_DecisionTree import information_gain

### 3. Create an object for Information Gain class
> ig = information_gain(X_train, y_train)

### 4. Add Column Features to the model
> ig.add_features(dataset, result_col_name)

### 5. Build the Decision Tree Model
> ig.decision_tree()

### 5. Predict the Test Set Results
> y_pred = ig.predict(X_test)

----------



Example Code
-------------

### 0. Download the dataset
Download dataset from [here](https://drive.google.com/file/d/1qjh3SnbrOY3ROXFYYMbJqQ7SvTbI6iqe/view?usp=sharing)

### 1. Import the dataset and Preprocess
> * import numpy as np
> * import matplotlib.pyplot as plt
> * import pandas as pd

> * dataset = pd.read_csv('house-votes-84.csv')
> * rawdataset = pd.read_csv('house-votes-84.csv')
> * party = {'republican':0, 'democrat':1}
> * vote = {'y':1, 'n':0, '?':0}

> * for col in dataset.columns:
>     * if col != 'party':
>         * dataset[col] = dataset[col].map(vote)
> * dataset['party'] = dataset['party'].map(party)

> * X = dataset.iloc[:, 1:17].values
> * y = dataset.iloc[:, 0].values

> * from sklearn.model_selection import KFold
> * kf = KFold(n_splits=5)

> * for train_index, test_index in kf.split(X,y):
>    * X_train, X_test = X[train_index], X[test_index]
>    * y_train, y_test = y[train_index], y[test_index]

### 2. Use the ID3 Library
> * from ID3 import information_gain
> * ig = information_gain(X_train, y_train)
> * ig.add_features(dataset, 'party')
> * print(ig.features)

> * ig.decision_tree()
> * y_pred = ig.predict(X_test)


----------



Footnotes
-------------

You can find the code at my [Github](https://github.com/safir72347/ML-ID3-Decision-Tree-Classification-Library-PyPi).



Connect with me on Social Media
-------------

* [https://www.github.com/safir72347](www.github.com/safir72347)
* [https://www.linkedin.com/in/safir72347/](https://www.linkedin.com/in/safir72347/)
* [https://www.instagram.com/safir_12_10/](https://www.instagram.com/safir_12_10/)