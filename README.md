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





## EDA

The data is imported as dataframe, and there is no missing value.

以敘述統計檢查資料分布狀況，
區分連續型數據跟離散型數據，
有六個連續型數據，
對除了time的六個連續型數據檢查是否為常態分布，並做偏態分析，

值在-1~1 之间为正常值， 正的是右偏，負的是左偏，
有三個偏態明顯的特徵值分別是
creatinine_phosphokinase : 4.46

serum_creatinine : 4.456

platelets : 1.462

接著分出連續型數據的dataframe實際來看一下連續型數據的分布情況

先看盒鬚圖，可以明顯看出creatinine_phosphokinase,serum_creatinine和platelets的離群值非常多，需要再對此三個特徵值做處理

接著看常態分佈圖，幾乎所有的數據都是有個高峰的，所以整體來說數據不用太多大處理

最後來看全部特徵的相對分布圖，可以看到全部特徵的散點圖跟直方圖


## 預處理

對數據的偏態進行處理，對於偏態較嚴重的reatinine_phosphokinase和serum_creatinine，我針對這兩個數據取了對數，對於偏態不大的platelets，我對這個數據做取平方根的處理


處理完後偏態數值
creatinine_phosphokinase log : 0.41400698865657504

serum_creatinine log : 1.583989782127556

platelets sqrt : 0.17868001456234672


Check the correlation coefficient of the target value with the heat map, then select the features.



## Training and Prediction

我刪除了三個特徵值，分別為刪掉 "diabetes","sex", "smoking" 為了減少電腦跑程式的負擔和增加準確度，結果證明準確度增加了0.05
then X and y are divided into training set and test set at a ratio of 8:2.

Create a Classification. Use sklearn's Decision Tree Classifier package to train the model. Then get the predicted model. Make predictions on testing data.

The prediction results show that precision and F1 score reaches 0.90.

## Create Plot

使用sklearn中的export_graphviz套件，做出決策樹圖表輸出dot檔案，並使用CMD輸入dot -Tpng pred.dot -o pred.png，將dot檔案轉為png檔，就能得到完整的決策樹圖表

再來就是針對建構好的模型和預測結果，對於特徵值的重要程度進行排序
最重要的特徵是
Feature: time, Score: 0.527

