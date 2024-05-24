import pandas as pd
import numpy as np

# Splitting into Test and Train Sets
from sklearn.model_selection import train_test_split


def split_data(X, y, test_size=0.2, random_state=42):
    """
        Parameters:
        - X: DataFrame containing the independent variables.
        - y: Dependent variable (target).
        - test_size: Proportion of the test set (default: 0.2).
        - random_state: State of randomness (default: 42).
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


############################################################
###################### Linear Regression ######################

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import numpy as np


def lin_model(X_train, y_train, X_test, y_test):
    """
        Trains, evaluates, and prints performance metrics for the Linear Regression model.

        Parameters:
        X_train, y_train: Training data and targets
        X_test, y_test: Testing data and targets

        Returns:
        reg_model: Trained regression model
    """

    # Creating and Training the Model
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)

    # Evaluating the Model's Performance & Prediction Accuracy
    train_score = reg_model.score(X_train, y_train)
    scores = cross_val_score(reg_model, X_train, y_train, cv=10)  # Evaluating the Model with Cross-Validation
    y_pred = reg_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # Square Root of MSE
    mae = mean_absolute_error(y_test, y_pred)
    reg_model.intercept_
    reg_model.coef_

    # Printing Performance Metrics
    print(f"Model skorları: {train_score}")
    print(f"Çapraz doğrulama skorları: {scores}")
    print(f"Ortalama skor: {scores.mean()}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Squared Error: {mae}")
    return reg_model


######################################################
# Linear Model Optimization

# Removing Low Coefficient Variables to Enhance Model Strength
#
# To prevent overfitting, if the model has too many features and some have very low (absolute value) or near zero coefficients,
# removing these features can reduce the model's complexity and improve its performance.
def optimize_model(X_train, y_train, X_test, y_test, threshold=0.02):
    """
        Trains a Linear Regression model using the given training and testing data,
        removes features with low impact, and retrains the model.

        Parameters:
        - X_train, y_train: Training data and targets
        - X_test, y_test: Testing data and targets
        - threshold: Threshold value for the absolute values of coefficients

        Returns:
        - reg_model: Retrained regression model
    """
    # Creating and Training the Model
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)

    # Placing Coefficients and Feature Names into a DataFrame
    coefficients = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': reg_model.coef_
    })

    # Filtering Low Coefficient Features: threshold = 0.02 (based on absolute values)
    low_coefficients = coefficients[np.abs(coefficients['Coefficient']) < threshold]
    print("Özellikler düşük katsayıya sahip (modelden çıkarılabilir):")
    print(low_coefficients)

    # Features with Low Coefficients (can be removed from the model)
    low_impact_features = low_coefficients['Feature'].tolist()
    X_train_dropped = X_train.drop(low_impact_features, axis=1)
    X_test_dropped = X_test.drop(low_impact_features, axis=1)

    # Retraining the Model
    reg_model.fit(X_train_dropped, y_train)

    # Evaluating the Model on Test Data
    train_score = reg_model.score(X_train_dropped, y_train)
    y_pred = reg_model.predict(X_test_dropped)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model skorları: {train_score}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R^2 Score: {r2}")

    return reg_model, X_train_dropped, X_test_dropped


############################################################
###################### Random Forests ######################
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def rf_model(X_train, y_train, X_test, y_test, threshold=0.02):
    """
        Trains, tests, and tunes hyperparameters for a Random Forest model.

        Parameters:
        X_train, y_train: Training data and targets
        X_test, y_test: Testing data and targets

        Returns:
        rf_tuned: Tuned and trained Random Forest model
    """

    # Creating and Training the Model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # # Placing Coefficients and Feature Names into a DataFrame
    # feature_importances  = pd.DataFrame({
    #     'Feature': X_train.columns,
    #     'Importance': rf_model.feature_importances_
    # })
    #
    # # Filtering Low Importance Features: threshold = 0.02 (based on absolute values)
    # low_importance_features = feature_importances[feature_importances['Importance'] < 0.02]['Feature'].tolist()
    # low_importance_features = feature_importances[np.abs(feature_importances['Importance']) < 0.005]
    # print("Özellikler düşük öneme sahip, modelden çıkarılabilir:")
    # print(low_importance_features)
    #
    # # Feature Removal Process
    # X_train_dropped = X_train.drop(low_importance_features, axis=1)
    # X_test_dropped = X_test.drop(low_importance_features, axis=1)

    # Retraining the Model
    rf_model.fit(X_train, y_train)

    # Prediction
    # train error
    y_pred = rf_model.predict(X_train)
    mse_train = np.sqrt(mean_squared_error(y_train, y_pred))
    print(f"Train Random Forest MSE: {mse_train}")

    # test error
    y_pred = rf_model.predict(X_test)
    mse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test Random Forest MSE: {mse_test}")

    # R^2 skoru
    r2 = r2_score(y_test, y_pred)
    print(f"R^2 Score: {r2}")
    # mae
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")

    # Model Tuning (Hiperparametre Tuning)
    rf_params = {
        # "max_features": [2, 5, 10],
        # "n_estimators": [200, 500],
        'max_depth': [2, 5, 10],  # Daha geniş bir aralıkta derinlik değerleri
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6, 10]}

    """
        # Limiting the max_depth value prevents trees from becoming too deep and overfitting.
        # Setting this value too low can lead to underfitting of the model.
        # The n_jobs parameter specifies the number of CPU cores to use for processing. If set to -1, all available processing cores are used.
    """

    # Creating a model with Grid Search
    rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_params, cv=5, n_jobs=-1, verbose=2,
                               scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    # grid_search.best_params_
    print(f"Best parameters: {grid_search.best_params_}")

    # Final Model
    rf_tuned = RandomForestRegressor(**grid_search.best_params_, random_state=42)
    rf_tuned.fit(X_train, y_train)
    y_pred_final = rf_tuned.predict(X_test)
    mse_final = np.sqrt(mean_squared_error(y_test, y_pred_final))
    print(f"Final model Random Forest MSE: {mse_final}")
    return rf_tuned, X_train, X_test


######################################################
###################### LightGBM ######################
from lightgbm import LGBMRegressor


def lgb_model(X_train, y_train, X_test, y_test):
    # Model
    lgb_model = LGBMRegressor().fit(X_train, y_train)
    y_pred = lgb_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    r2 = r2_score(y_test, y_pred)

    # Model Tuning
    lgb_model = LGBMRegressor()
    lgbm_params = {"learning_rate": [0.01, 0.05, 0.1],  # Determines the learning rate of the model
                   "n_estimators": [100, 500, 1000],  # Determines the number of trees to be used in the model
                   "max_depth": [3, 5, 8],  # Determines the maximum depth for each tree
                   "colsample_bytree": [1, 0.8, 0.5]}  # Determines the fraction of features to be used for each tree

    # GridSearchCV performs cross-validation to find the best parameters that yield the best results within the specified parameter ranges.

    lgbm_cv_model = GridSearchCV(lgb_model,
                                 lgbm_params,
                                 cv=10,
                                 n_jobs=-1,
                                 verbose=2).fit(X_train, y_train)

    print(f"Best parameters: {lgbm_cv_model.best_params_}")

    # Final Model
    lgbm_tuned = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X_train, y_train)
    y_pred_tuned = lgbm_tuned.predict(X_test)
    tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_tuned = r2_score(y_test, y_pred_tuned)
    print(f"Light GBM Mean Square Error (train): {mse}")
    print(f"Root Mean Square Error (train): {rmse}")
    print(f"R^2 Score: {r2}")
    print(f"Tuned RMSE: {tuned_rmse}")
    print(f"Tuned Model R^2 Score: {r2_tuned}")
    return lgbm_tuned, rmse, tuned_rmse, r2
    # lgbm_tuned, X_train, X_test)
