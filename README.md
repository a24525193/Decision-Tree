# Decision-Tree
Pattern Recognition Assignment 2 using Python


## Data Set

https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records

This dataset contains the medical records of 299 patients who had heart failure, collected during their follow-up period, where each patient profile has 13 clinical features.

Attribute Information:

- age: age of the patient (years)
- anaemia: decrease of red blood cells or hemoglobin (boolean)
- high blood pressure: if the patient has hypertension (boolean)
- creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)
- diabetes: if the patient has diabetes (boolean)
- ejection fraction: percentage of blood leaving the heart at each contraction (percentage)
- platelets: platelets in the blood (kiloplatelets/mL)
- sex: woman or man (binary)
- serum creatinine: level of serum creatinine in the blood (mg/dL)
- serum sodium: level of serum sodium in the blood (mEq/L)
- smoking: if the patient smokes or not (boolean)
- time: follow-up period (days)
- [target] death event: if the patient deceased during the follow-up period (boolean)



## Package used

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from math import exp
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.tree import export_graphviz
from matplotlib import pyplot
```





## Exploratory Data Analysis

The data is imported as dataframe, and there is no missing value.

Check the distribution of data with descriptive statistics.
Distinguish between continuous data and discrete data.

We found seven continuous data in the dataset.
Check whether the six continuous data except 'time' are normally distribution. Then I conduct skewness analysis.

The skewness between -1 and 1 is in the normal value. The positive skewness is right skewness, and the negative skewness is left skewness.
There are three obvious characteristic values of skewness, which are

**creatinine_phosphokinase : 4.46**

**serum_creatinine : 4.456**

**platelets : 1.462**



Next, separate the dataframe of continuous data. Let's actually see the distribution of continuous data.


First, look at the box-and-whisker plot. We can clearly see that There are many outliers in 'creativity_phosphokinase', 'serum_creatinine' and 'platelets', so we need to deal with these three features.


Then look at the normal distribution plot. Almost all data centers have peaks, so the data need not be processed too much overall.



Finally, look at the relative distribution map of all features. You can see the scatter plot and histogram of all features.


## Preprocessing

To deal with the skewness of the data. For 'reatinine_phosphokinase' and 'serum_creatinine' with severe skewness, I took the logarithms of these two data.
And for 'platelets' with little skewness, I took the square root of this data.


Skewed values after processing

creatinine_phosphokinase log : 0.41400698865657504

serum_creatinine log : 1.583989782127556

platelets sqrt : 0.17868001456234672

I also show the results of skewness processing by outputting histograms plots.

And I check the correlation coefficient of the target value with the heat map, then select the features.



## Training and Prediction

I deleted the three features whose correlation coefficient with the target value 'DEATH_EVENT' are lower than 0.02. 

They are "diabetes", "sex", "smoking". In order to reduce the burden of running the program on the computer and increase the accuracy.

Set the other 9 features to X and 'DEATH_EVENT' to Y. Then X and y are divided into training set and test set at a ratio of 8:2.

Create a Classification. Use sklearn's Decision Tree Classifier package to train the model. 

Set model parameters. The parameter meanings of the function are as follows:

- max_Depth: int or None, optional (default=None) Sets the maximum depth of the decision tree in the decision random forest. The greater the depth, the easier it is to over fit. The recommended depth of the tree is 5-20.
- max_Features: None, log2, sqrt, when the N feature is less than 50, all
- Criterion: gini or entropy, the former is Gini coefficient, and the latter is information entropy.
- min_samples_Leaf: This value limits the minimum number of samples of a leaf node. If the number of a leaf node is less than the number of samples, it will be pruned together with its sibling nodes.
- min_samples_Split: Set the minimum number of samples for the node. When the number of samples may be less than this value, the node will not be divided.
- random_state: A random seed, which is used as a parameter in any random class or function to control the random pattern.

Then get the predicted model. Make predictions on testing data.

The prediction results show that precision and F1 score reaches 0.90.

## Create Plot

Use sklearn's export_graphviz package. Make a decision tree plot of the model. And output a dot file.
Then use CMD to input the following code.
```dot -Tpng pred.dot -o pred.png```

Convert the dot file to the png file to get a complete decision tree plot.

The next step is to rank the importance of features according to the constructed model and prediction results.

The most important feature is 'time'. Important score: 0.529 (Max : 1)

