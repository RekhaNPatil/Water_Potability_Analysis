import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



water_data = pd.read_csv('water_potability ml project dataset.csv')
print(water_data)
water_data.head()

print(water_data.columns)

print(water_data.dtypes)

print(water_data.info())
# missing values

print(water_data.describe())

print(water_data.duplicated().any())

print(water_data.isnull().sum())
# null values present

# percentage of missing values
null_df=water_data.isnull().sum().reset_index()
null_df.columns=['column','Null_count']
null_df['%miss_value']=round(null_df['Null_count']/len(water_data),2)*100
print(null_df)

# Handling Missing Values
# filling the missing values by mean
water_data['ph']=water_data['ph'].fillna(water_data['ph'].mean())
water_data['Trihalomethanes']=water_data['Trihalomethanes'].fillna(water_data['Trihalomethanes'].mean())
water_data['Sulfate']=water_data['Sulfate'].fillna(water_data['Sulfate'].mean())

print(water_data.isnull().sum())

# Check for Correlation
corr_matrix=water_data.corr()
print(corr_matrix)

plt.figure(figsize=(18, 16))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.show()

data_hist_plot = water_data.hist(figsize=(20, 20), color="#5F9EA0")
# water_data.show()

for col in water_data.columns:
  sns.histplot(data=water_data, x=col, kde=True, hue='Potability')
# plt.show()

print(water_data['Potability'].value_counts())

sns.countplot(water_data['Potability'])
water_data['Potability'].value_counts().plot(kind='bar')

x = water_data.drop('Potability',axis=1)
y = water_data['Potability']
# print
print(x.head())

print(y.head())# dependent variable or target variable

# Feature Scalling
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()

x_scaled=std_scaler.fit_transform(x)
print(x_scaled)

# Training and Testing Dataset:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42,stratify=y)

x_train.shape,x_test.shape

# importing models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Creating the object of the model:
LR = LogisticRegression()
DT = DecisionTreeClassifier()
RF = RandomForestClassifier()
KNN = KNeighborsClassifier()
NB = GaussianNB()

from sklearn.model_selection import cross_val_score
models = [LR, DT, RF, KNN, NB]
features = x_scaled
labels = y
CV = 5
accu_list = []# Accuracy List
ModelName = []# Model Name List

for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model,features,labels,scoring='accuracy',cv=CV)
  accu_list.append(accuracies.mean()*100)
  ModelName.append(model_name)

model_acc_df = pd.DataFrame({"Model":ModelName,"Cross_Val_Accuracy":accu_list})
print(model_acc_df)

from sklearn.metrics import classification_report
RF.fit(x_train, y_train)
y_pred_rf = RF.predict(x_test)

#Random Forest
print(classification_report(y_test, y_pred_rf))

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

params_RF={"min_samples_split":[2,6],
           "min_samples_leaf":[1,4],
           "n_estimators":[100,200,300],
           "criterion":["gini",'entropy']
           }

cv_method = StratifiedKFold(n_splits=3)
GridSearchCV_RF = GridSearchCV(estimator=RandomForestClassifier(),
                               param_grid=params_RF,
                               cv=cv_method,
                               verbose=1,
                               n_jobs=2,
                               scoring="accuracy",
                               return_train_score=True
                               )

GridSearchCV_RF.fit(x_train, y_train)
best_params_RF = GridSearchCV_RF.best_params_
print("Best Hyperparameters for Random Forest are=",best_params_RF)

best_estimator = GridSearchCV_RF.best_estimator_
best_estimator.fit(x_train, y_train)
y_pred_best = best_estimator.predict(x_test)
print(classification_report(y_test, y_pred_best))

from sklearn.metrics import accuracy_score
print(f"Accuracy of Random Forest Model={round(accuracy_score(y_test,y_pred_best)*100, 2)}%")

# Predictive System
list1 = water_data.iloc[2:3,0:9].values.flatten().tolist()
print(list1)

pickle.dump(best_estimator,open("Model.pkl","wb"))
pickle.dump(std_scaler,open("scalar.pkl","wb"))


