import pandas as pd
import numpy as np
pd.pandas.set_option('display.max_columns', None)
from models.models import *  # Module Containing the Models

###################### Data Preparation (EDA) ######################
from helpers.helpers import *  # Module Prepared for Exploratory Data Analysis (EDA)

# from helpers.helpers import check_df, cat_summary

dfMat = pd.read_csv(r"student\resources\student-mat.csv", sep=";")
dfPor = pd.read_csv(r"student\resources\student-por.csv", sep=";")

dfMat.describe()
# dfMat.columns

### Missing Values ###
check_df(dfMat)  # NA yok
check_df(dfPor)  # NA yok

cat_summary(dfMat, "reason")
cat_summary(dfMat, "sex", True)
cat_summary(dfMat, "famsize")
cat_summary(dfMat, "Mjob")
cat_summary(dfMat, "guardian")
cat_summary(dfMat, "health")

### Outliers ###
from helpers.helpers import check_outlier, grab_outliers, replace_with_thresholds

# visulazitation for Outliers
sns.boxplot(x=dfMat["absences"])
plt.show()

# check Outliers
check_outlier(dfMat, "G1")
check_outlier(dfMat, "G2")
check_outlier(dfMat, "absences")

######################################################
# Clamping outliers to lower and upper values
grab_outliers(dfMat, "G2")
replace_with_thresholds(dfMat, "G2")
grab_outliers(dfMat, "absences")
replace_with_thresholds(dfMat, "absences")

######################################################
# Label Encoding & One Hot Encoding
# ohe_cols = [col for col in dfMat.columns if 10 >= len(dfMat[col].unique()) > 2]

# one hot encoding >> school famsize Mjob Fjob reason guardian
ohe_cols = ['school', 'famsize', 'Mjob', 'Fjob', 'reason', 'guardian']  # ordinal değişkenler değil
# dfMat[['school', 'famsize', 'Mjob', 'Fjob', 'reason', 'guardian']]
df = one_hot_encoder(dfMat, ohe_cols, drop_first=True)

binary_cols = [col for col in df.columns if df[col].dtypes == "O"
               and len(df[col].unique()) == 2]

for col in binary_cols:
    label_encoder(df, col)

# Feature Engineering
df['Avg_G'] = (df['G1'] + df['G2']) / 2
df['Change_G'] = df['G2'] - df['G1']

######################################################
# Splitting the Dataset into Training and Testing Sets
X = df.drop('G3', axis=1)
y = df['G3']

X_train, X_test, y_train, y_test = split_data(X, y)

# Correlation Matrix
# A heatmap shows the relationships & correlations between variables.

import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = X_train.corr()

plt.figure(figsize=(30, 30))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Korelasyon Matrisi')
# plt.savefig('korelasyon_matrisi.png')
plt.show()

######################################################
# Linear Regression Model, Function Call
# Before removing any variables from the model.

reg_model = lin_model(X_train, y_train, X_test, y_test)

######################################################
# Model (Final model)
# modeli optimize edilmiş ve güncellenmiş veri setlerini alma
reg_optimized_model, X_train, X_test = optimize_model(X_train, y_train, X_test, y_test)

###################### Random Forests ######################
X_train, X_test, y_train, y_test = split_data(X, y)
# Final Model (Random Forest)
rf_model = rf_model(X_train, y_train, X_test, y_test)

###################### LightGBM ######################
X_train, X_test, y_train, y_test = split_data(X, y)
# Final Model (LightGBM)
lgb_model = lgb_model(X_train, y_train, X_test, y_test)

######################################################
# Feature Importance & visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_importance(model_info, X, num=len(X)):
    # Check if model_info is a tuple and extract the model
    if isinstance(model_info, tuple):
        model = model_info[0]  # Adjust index if model is not the first element
    else:
        model = model_info

    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': X.columns})
    plt.figure(figsize=(10, 15))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Feature Importance')
    plt.tight_layout()
    # plt.savefig('feature importance.png')
    plt.show()


# visualization feature importance
plot_importance(rf_model, X)
plot_importance(lgb_model, X)

# Regresyon grafiğini çizdirme
score = X_train['G2']

g = sns.regplot(x=score, y=y_train, scatter_kws={'color': 'b', 's': 9}, ci=False, color="r")
g.set_title(f"Model Denklemi: StuPerformance = {round(reg_model.intercept_, 2)} + {round(reg_model.coef_[0], 2)}*Score")
g.set_ylabel("StuPerformance")
g.set_xlabel("Score")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()

######################################################
######################################################
# student-por.csv - dfPor
dfPor = pd.read_csv(r"student\resources\student-por.csv", sep=";")
dfPor.describe()

### Missing Values ###
check_df(dfPor)  # NA yok

# cat_summary(dfPor, "reason")
# cat_summary(dfPor, "sex", True)
# cat_summary(dfPor, "famsize")
# cat_summary(dfPor, "Mjob")
# cat_summary(dfPor, "guardian")
# cat_summary(dfPor, "health")

### Outliers ###
sns.boxplot(x=dfMat["absences"])
plt.show()

# check Outliers
check_outlier(dfPor, "G1")
check_outlier(dfPor, "G2")
check_outlier(dfPor, "absences")

# Clamping outliers to lower and upper values
grab_outliers(dfPor, "G1")
replace_with_thresholds(dfPor, "G1")
grab_outliers(dfPor, "G2")
replace_with_thresholds(dfPor, "G2")
grab_outliers(dfPor, "absences")
replace_with_thresholds(dfPor, "absences")

######################################################
# Label Encoding & One Hot Encoding
ohe_cols = ['school', 'famsize', 'Mjob', 'Fjob', 'reason', 'guardian']
df = one_hot_encoder(dfPor, ohe_cols, drop_first=True)

binary_cols = [col for col in df.columns if df[col].dtypes == "O"
               and len(df[col].unique()) == 2]

for col in binary_cols:
    label_encoder(df, col)

######################################################
# Feature Engineering
df['Avg_G'] = (df['G1'] + df['G2']) / 2
df['Change_G'] = df['G2'] - df['G1']

######################################################
# Splitting the Dataset into Training and Testing Sets

X = df.drop('G3', axis=1)
y = df['G3']

X_train, X_test, y_train, y_test = split_data(X, y)

# Correlation Matrix
corr_matrix = X_train.corr()

plt.figure(figsize=(30, 30))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Korelasyon Matrisi')
# plt.savefig('korelasyon_matrisi_por.png')
plt.show()

######################################################
X_train, X_test, y_train, y_test = split_data(X, y)
# Final Model (Linear Regression)
reg_model = lin_model(X_train, y_train, X_test, y_test)

# Optimizing the Model and Obtaining Updated Datasets
reg_optimized_model, X_train, X_test = optimize_model(X_train, y_train, X_test, y_test)

######################################################
# Final Model (Random Forest)
rf_model = rf_model(X_train, y_train, X_test, y_test)

###################### LightGBM ######################
# Final Model (LightGBM)
lgb_model = lgb_model(X_train, y_train, X_test, y_test)
######################################################

# Model save
import pickle

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


# save_model(optimized_model, 'linear_model_mat.pkl')
# save_model(rf_model, 'randomForest_model_mat.pkl')
# save_model(optimized_model, 'linear_model_por.pkl')
# save_model(rf_model, 'randomForest_model_por.pkl')
# save_model(lgb_model, 'lightgbm_model_mat.pkl')


with open("student\resources\linear_model_mat.pkl", "wb") as f:
    pickle.dump(reg_optimized_model, f)

with open("student\resources\randomForest_model_mat.pkl", "wb") as f:
    pickle.dump(rf_model, f)

with open("student\resources\rlightgbm_model_mat.pkl", "wb") as f:
    pickle.dump(rf_model, f)

X_train.head(1)

######################################################
# Sample data / Linear Regression Model
X_test_sample = np.array([[
    1, 16, 1, 1, 3, 1, 2, 0,  # sex, age, address, Pstatus, Fedu, traveltime, studytime, failures
    0, 0, 1, 1, 1, 1, 1, 1,  # schoolsup, famsup, paid, activities, nursery, higher, internet, romantic
    4, 3, 1, 2, 3, 2, 12, 13,  # famrel, goout, Dalc, Walc, health, absences, G1, G2
    0, 0, 0, 1, 0, 0, 1, 0,
    # school_MS, Mjob_health, Mjob_other, Mjob_services, Mjob_teacher, Fjob_health, Fjob_other, Fjob_teacher
    1, 0, 0, 1, 0, 12.5, 1
    # reason_home, reason_other, reason_reputation, guardian_mother, guardian_other, Avg_G, Change_G
]])

########### Linear Regression Model #############
# Making Predictions with the Linear Regression Model
y_test_pred_new = reg_optimized_model.predict(X_test_sample)
print("Linear Regression test verisi için tahmin edilen değer:", y_test_pred_new)

########### Random Forest Model #############
model = rf_model[0]

# Making Predictions with the Random Forest Model
y_test_pred_new = model.predict(X_test_sample)
print("Random Forest test verisi için tahmin edilen değer:", y_test_pred_new)
