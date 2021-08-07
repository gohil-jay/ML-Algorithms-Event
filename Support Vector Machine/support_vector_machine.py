import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Assessing the Dataset

cancer_data = datasets.load_breast_cancer()
dataset = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
dataset['target'] = cancer_data.target
dataset.head()
dataset.tail()
dataset

# Understanding the dataset

#Printing dataframe information
print("Dataset Information -->\n")
print(dataset.info())
print("\n"*2)

#Printing target values of dataset
target = cancer_data.target
print("All target values -->\n")
print(target)
print("\n"*2)

#Printing unique target values of dataset
unique = dataset['target'].unique()
print("Unique target values -->\n")
print(unique)
print("\n"*2)

#Checking the presence of NULL values
null_check = dataset.isnull().sum()
print("Null values -->\n")
print(null_check)
print("\nAs clearly seen above, there are no null values and thus, there's no need to process it.")
print("\n"*2)

#Printing mathematical dataset analysis
print("Rigorous mathematical analysis of dataset -->\n")
print(dataset.describe())

# Identifying necessary variables

# Understanding correlation among columns
print("Visualizing correlation between features using heatmap -->", "\n"*2)
fig, ax = plt.subplots(figsize=(10,10))
temp_plot = sns.heatmap(dataset.corr(method='pearson'), cmap='Reds', ax=ax)
print(temp_plot)

# Engineering the data
x = dataset.iloc[:, 0:30]
y = dataset[["target"]]
print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

# Performing Regression
model = svm.SVC(kernel='linear')
model.fit(x_train,y_train)

# Predicting Test Values

y_pred = model.predict(x_test)
print(y_pred)
print(y_test)

# Evaluating model

acc = accuracy_score(y_pred, y_test)
print("Accuracy : ", acc)
mse = mean_squared_error(y_pred,y_test)
print("Mean Square Error : ", mse)

# Visualizing model

temp = 10
original_values = y_test.iloc[0:temp].to_numpy()
predicted_values = y_pred[0:temp]
list_values = []
for i in range(temp):
  list_values.append(i)
plt.scatter(list_values, predicted_values, marker="+", color = 'orange')
plt.scatter(list_values, original_values, color = 'blue')

# Custom Prediction

custom_pred = model.predict([[17.99, 10.38, 122.80, 1001.0, 0.11840, 0.27760, 0.3001, 0.14710, 0.2419, 0.07871, 1.0950, 0.9053, 8.589, 153.40, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.60, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.11890]])
print("Prediction for custom aforementioned values : ", custom_pred[0])

# Thank you!
