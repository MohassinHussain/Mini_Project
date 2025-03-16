import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

st.title("Rainfall Analysis and Flood Prediction")

file_path = 'rainfall.csv'  
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()

data['rainfall'] = data['rainfall'].map({'yes': 1, 'no': 0})

numerical_features = ['temparature', 'humidity', 'windspeed', 'pressure', 'intensity']

numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_features)
])

X = data[numerical_features]
y = data['intensity']

X_preprocessed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

regressor = RandomForestRegressor(n_estimators=200, random_state=42)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

st.subheader("Rainfall Intensity Prediction")
st.write(f"Mean Absolute Error: `{mae:.4f}`")

X_flood = data[['rainfall', 'intensity', 'temparature', 'humidity', 'windspeed', 'pressure']]
y_flood = data['flood_occurence']

X_flood_train, X_flood_test, y_flood_train, y_flood_test = train_test_split(X_flood, y_flood, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_flood_train, y_flood_train)

best_model = grid_search.best_estimator_
y_flood_pred = best_model.predict(X_flood_test)
flood_accuracy = accuracy_score(y_flood_test, y_flood_pred)

st.subheader("Flood Prediction Model")
st.write(f"Best Model Accuracy: `{flood_accuracy:.4f}`")

advice = []
for _, row in data.iterrows():
    if row['intensity'] > 7.6:
        advice.append("High rainfall, consider drainage systems")
    elif row['intensity'] < 7 and row['intensity'] < 2.5:
        advice.append("Moderate rainfall, suitable for most crops")
    else:
        advice.append("Low rainfall, consider irrigation techniques")

data['advisory'] = advice

output_file = 'rainfall_analysis_output.csv'
data.to_csv(output_file, index=False)

st.subheader("Rainfall Intensity Distribution")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data['intensity'], bins=30, kde=True, color='blue', ax=ax, label="Rainfall Intensity Distribution")
plt.xlabel("Rainfall Intensity")
plt.ylabel("Frequency")
plt.legend()
st.pyplot(fig)

st.subheader("Flood Risk Based on Rainfall")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x=data['rainfall'], y=data['flood_occurence'], ax=ax)
plt.xlabel("Rainfall (0 = No, 1 = Yes)")
plt.ylabel("Flood Occurrence Risk")
plt.legend(["Flood Occurrence"])
plt.grid(True)
st.pyplot(fig)

st.subheader("Rainfall Intensity vs. Temperature Regression")
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(x=data['temparature'], y=data['intensity'], scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
plt.xlabel("Temperature (°C)")
plt.ylabel("Rainfall Intensity")
plt.title("Regression: Rainfall Intensity vs. Temperature")
plt.legend(["Regression Line"])
st.pyplot(fig)