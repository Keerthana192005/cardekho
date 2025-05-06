import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder

# Load the dataset
df = pd.read_csv('cardekho_imputated.csv', index_col=0)

# 'car_name' column is not present, so remove the drop statement
# df.drop('car_name', axis=1, inplace=True)  # Remove this line


# Apply Label Encoding to the 'model' column (or the identified problematic column)
le = LabelEncoder()
df['model'] = le.fit_transform(df['model']) # Change the column as required.

#Apply one hot encoding

df = pd.get_dummies(df, drop_first=False)  #Set drop_first = False to resolve your error

# Separate features (X) and target (y)
X = df.drop('selling_price', axis=1)
y = df['selling_price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')

# Prepare new data for prediction
# ... (your new_data DataFrame) ...
new_data = pd.DataFrame({
    'vehicle_age': [3, 6],
    'km_driven': [25000, 60000],
    'mileage': [20.5, 15.8],
    'engine': [1200, 1500],
    'max_power': [85, 120],
    'seats': [5, 5],
    'brand_Audi': [1, 0],
    'brand_BMW': [0, 1],
    'brand_Chevrolet': [0, 0],
    'brand_Ford': [0, 0],
    'brand_Honda': [0, 0],
    'brand_Hyundai': [0, 0],
    'brand_Kia': [0, 0],
    'brand_Maruti': [0, 0],
    'brand_Mercedes-Benz': [0, 0],
    'brand_Nissan': [0, 0],
    'brand_Renault': [0, 0],
    'brand_Tata': [0, 0],
    'seller_type_Individual': [1, 1],
    'seller_type_Trustmark Dealer': [0, 0],
    'fuel_type_Diesel': [0, 1],
    'fuel_type_Electric': [0, 0],
    'fuel_type_LPG': [0, 0],
    'fuel_type_Petrol': [1, 0],
    'transmission_type_Manual': [1, 1]
})

# Get missing columns from training data
missing_cols = set(X_train.columns) - set(new_data.columns)

# Add missing columns to new_data and fill with 0
for col in missing_cols:
    new_data[col] = 0

# Reorder columns to match training data
new_data = new_data[X_train.columns]

# Make predictions on the new data
new_predictions = model.predict(new_data)
print("Predicted prices for the new data:")
print(new_predictions)