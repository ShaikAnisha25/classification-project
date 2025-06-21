
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# Load data
df = pd.read_excel('pima-data.xlsx')

# Display entire dataframe
pd.set_option('display.max_rows', None)
print(df)
pd.reset_option("display.max_rows")

# Shape of dataset
print(df.shape)

# Check for missing values
print(df.isnull().values.any())

# Correlation matrix
corr = df.corr()
fig, ax = plt.subplots(figsize=(12, 12))
cmap = 'plasma'
ax.matshow(corr, cmap=cmap)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()

# Drop unnecessary columns
df.drop(['thickness', 'has_diabetes', 'diabetes_orig'], axis=1, inplace=True)

# Replace True/False with 1/0
df.replace({'diabetes': {True: 1, False: 0}}, inplace=True)

# Display first few rows
print(df.head())

# Count values
num_true = len(df.loc[df['diabetes'] == 1])
num_false = len(df.loc[df['diabetes'] == 0])
print(f'num_true = {num_true}')
print(f'num_false = {num_false}')

per_num_true = (num_true / (num_true + num_false)) * 100
print(per_num_true)
per_num_false = (num_false / (num_false + num_true)) * 100
print(per_num_false)

# Split data
input_columns = ['num_preg', 'glucose_conc', 'diastolic_bp', 'insulin', 'bmi', 'diab_pred', 'age', 'skin']
output_columns = ['diabetes']

x = df[input_columns].values
y = df[output_columns].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Class distribution
print('{0:0.2f}% in training set'.format((len(x_train) / len(df.index)) * 100))
print('{0:0.2f}% in testing set'.format((len(x_test) / len(df.index)) * 100))

print('training true: {} ({:.2f}%)'.format(len(y_train[y_train[:] == 1]), len(y_train[y_train[:] == 1]) / len(y_train) * 100))
print('training false: {} ({:.2f}%)'.format(len(y_train[y_train[:] == 0]), len(y_train[y_train[:] == 0]) / len(y_train) * 100))
print('Test true: {} ({:.2f}%)'.format(len(y_test[y_test[:] == 1]), len(y_test[y_test[:] == 1]) / len(y_train) * 100))
print('Test false: {} ({:.2f}%)'.format(len(y_test[y_test[:] == 0]), len(y_test[y_test[:] == 0]) / len(y_train) * 100))

# Missing values
cols_to_check = ['num_preg', 'glucose_conc', 'diastolic_bp', 'insulin', 'bmi', 'diab_pred', 'age']
for col in cols_to_check:
    total = len(df.loc[df[col] == 0])
    print(f'Number of rows missing in {col}: {total}')

# Handle missing values
fill_zeros = SimpleImputer(missing_values=0, strategy='mean')
x_train = fill_zeros.fit_transform(x_train)
x_test = fill_zeros.transform(x_test)

# GaussianNB
nb_model = GaussianNB()
nb_model.fit(x_train, y_train.ravel())
nb_predict_train = nb_model.predict(x_train)
print(f'Training Accuracy (NB): {metrics.accuracy_score(y_train, nb_predict_train)}')
nb_predict_test = nb_model.predict(x_test)
print(f'Test Accuracy (NB): {metrics.accuracy_score(y_test, nb_predict_test)}')
print(metrics.confusion_matrix(y_test, nb_predict_test))
print(metrics.classification_report(y_test, nb_predict_test))

# RandomForest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train, y_train.ravel())
rf_predict_train = rf_model.predict(x_train)
print(f'Training Accuracy (RF): {metrics.accuracy_score(y_train, rf_predict_train)}')
rf_predict_test = rf_model.predict(x_test)
print(f'Test Accuracy (RF): {metrics.accuracy_score(y_test, rf_predict_test)}')
print(metrics.confusion_matrix(y_test, rf_predict_test))
print(metrics.classification_report(y_test, rf_predict_test))

# KNeighbors
k_neighbor = KNeighborsClassifier()
k_neighbor.fit(x_train, y_train.ravel())
x_pred_knn = k_neighbor.predict(x_test)
print(f'Accuracy (KNN): {metrics.accuracy_score(y_test, x_pred_knn)}')
print(metrics.confusion_matrix(y_test, x_pred_knn))

# SVM
support = SVC()
support.fit(x_train, y_train.ravel())
x_pred_svm = support.predict(x_test)
print(f'Accuracy (SVM): {metrics.accuracy_score(y_test, x_pred_svm)}')
print(metrics.confusion_matrix(y_test, x_pred_svm))

# StandardScaler
scaler = StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)
print(standardized_data)

# Test new input data
input_data = [8, 183, 64, 0, 23.3, 0.672, 32, 0.0000]
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
std_data = scaler.transform(input_data_as_numpy_array)
support_model_test = support.predict(std_data)
print(support_model_test)
