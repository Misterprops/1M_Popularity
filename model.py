from flask import Flask, request, Response
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import os

app = Flask(__name__)

# Load the CSV file
global df
global xgb_model_corr
global comparison_df

CORS(app)  # Esto permite solicitudes desde cualquier origen
@app.route('/api/columns')
def columns():
    global df
    df = pd.read_csv('spotify_data.csv')
    modelo()
    return df.columns.values.tolist()

@app.route('/api/info', methods=['POST'])
def info():
    global comparison_df
    data = request.get_json()
    unique_data = comparison_df[f"Actual_{(data.get('columna'))}"].unique().tolist()
    return unique_data


def modelo():
    global df
    global comparison_df
    global xgb_model_corr
    
    categorical_mappings = {}

    # Identify columns that are currently objects (original categorical columns)
    object_cols_for_mapping = df.select_dtypes(include='object').columns.tolist()

    print("Saving mappings for original object columns...")
    for col in object_cols_for_mapping:
        # Convert to category temporarily to access the mapping
        temp_cat_col = df[col].astype('category')

        # Create a mapping from numerical code to original category name
        # The index of categories corresponds to the numerical codes
        mapping = dict(enumerate(temp_cat_col.cat.categories))

        # Store the mapping
        categorical_mappings[col] = mapping
        print(f"Mapping saved for column: {col}")

    for col in df.columns:
    # Check if the column's dtype is 'object'
        if df[col].dtype == 'object':
            # If it's an object type, convert it to a categorical type and then to numerical codes
            # Note: We've already saved the mapping, so we just do the conversion now
            df[col] = df[col].astype('category').cat.codes


    # Display the information of the modified DataFrame to verify the changes
    #df.info()

    X = df.drop('popularity', axis=1)
    y = df['popularity']

    # Separate the target variable (popularity) from the PCA components
    # Since df_pca only contains the PCA components, we need to use the original y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    

    # Assuming X_train, X_test, y_train, and y_test are already defined from the previous split

    # Calculate the correlation of each feature with 'popularity' using the training data
    # We calculate absolute correlation to consider strong negative correlations as well
    correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)

    # Select the top 10 feature names based on correlation
    # Exclude 'popularity' itself if it somehow ended up in X_train (it shouldn't based on how X was created)
    # Let's take the top 10 features from the correlation list
    features_corr = correlations.head(17).index.tolist()

    # Filter the training and testing dataframes to include only the top 10 features
    X_train_corr = X_train[features_corr]
    X_test_corr = X_test[features_corr]


    # Initialize the final XGBoost Regressor model with top 10 features selected by correlation
    xgb_model_corr = xgb.XGBRegressor(objective='reg:squarederror', # Objective for regression tasks
                                    n_estimators=1000,          # Number of boosting rounds
                                    learning_rate=0.1,         # Step size shrinkage
                                    max_depth=7,                # Maximum depth of a tree
                                    random_state=42,
                                    n_jobs=-1,
                                    reg_alpha=0.15,  # Add L1 regularization
                                    reg_lambda=0.15)

    # Train the final model using the top 10 features selected by correlation
    print("\nStarting XGBoost model training...")
    xgb_model_corr.fit(X_train_corr, y_train)
    print("XGBoost model training finished.")
    
    comparison_df = pd.DataFrame()

    for categorico in categorical_mappings:
        # Get the numerical genre codes for the rows in comparison_df from X_test_corr
        numerical_genre_codes = X_test_corr[categorico]

        # Create a Series to map the numerical codes back to original names
        # The index of this Series should be the numerical codes, and values should be original names
        genre_code_to_name = pd.Series(categorical_mappings[categorico])

        # Map the numerical codes in X_test_corr['genre'] back to original names
        original_genre_names = numerical_genre_codes.map(genre_code_to_name)

        # Add this as a new column to the comparison_df for the displayed head
        comparison_df[f'Actual_{categorico}'] = original_genre_names

    # Predict on the test set using the top 10 features selected by correlation
    y_pred_xgb_corr = xgb_model_corr.predict(X_test_corr)

    # Evaluate the model
    mse_xgb_corr = mean_squared_error(y_test, y_pred_xgb_corr)
    rmse_xgb_corr = np.sqrt(mse_xgb_corr)
    r2_xgb_corr = r2_score(y_test, y_pred_xgb_corr)

    print(f'\nXGBoost Model Performance:')
    print(f'Mean Squared Error (MSE): {mse_xgb_corr}')
    print(f'Root Mean Squared Error (RMSE): {rmse_xgb_corr}')
    print(f'R-squared (R2): {r2_xgb_corr}')

    print("\n--- Model Performance Comparison ---")
    print(f"Actual Popularity (first 5): {y_test[:20].values}")
    print(f"Predicted Popularity (first 5): {y_pred_xgb_corr[:20]}")
    
    return Response(status=204)
    
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render define PORT
    app.run(host="0.0.0.0", port=port)