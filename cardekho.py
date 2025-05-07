import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('cardekho_imputated.csv', index_col=0)

# Label encode 'model'
le = LabelEncoder()
df['model'] = le.fit_transform(df['model'])

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=False)

# Features and target
X = df.drop('selling_price', axis=1)
y = df['selling_price']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Get all columns for one-hot encoding
all_columns = X_train.columns.tolist()

# --- Streamlit App ---
st.title("Car Price Predictor 🚗")
st.write("Enter the details below to predict the car's selling price:")

# Numeric Inputs
vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=50, value=5)
km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=30000)
mileage = st.number_input("Mileage (km/l)", min_value=0.0, max_value=50.0, value=18.0)
engine = st.number_input("Engine Capacity (cc)", min_value=500, max_value=5000, value=1200)
max_power = st.number_input("Max Power (bhp)", min_value=20.0, max_value=500.0, value=80.0)
seats = st.number_input("Number of Seats", min_value=2, max_value=10, value=5)

# Dropdown Inputs
brand_options = [col.split('_')[1] for col in all_columns if col.startswith('brand_')]
brand = st.selectbox("Brand", sorted(set(brand_options)))

seller_options = [col.split('_')[2] if len(col.split('_')) > 2 else col.split('_')[1] for col in all_columns if col.startswith('seller_type_')]
seller_type = st.selectbox("Seller Type", sorted(set(seller_options)))

fuel_options = [col.split('_')[2] if len(col.split('_')) > 2 else col.split('_')[1] for col in all_columns if col.startswith('fuel_type_')]
fuel_type = st.selectbox("Fuel Type", sorted(set(fuel_options)))

trans_options = [col.split('_')[2] if len(col.split('_')) > 2 else col.split('_')[1] for col in all_columns if col.startswith('transmission_type_')]
transmission_type = st.selectbox("Transmission Type", sorted(set(trans_options)))

# Prepare user input dictionary
user_input = {
    'vehicle_age': vehicle_age,
    'km_driven': km_driven,
    'mileage': mileage,
    'engine': engine,
    'max_power': max_power,
    'seats': seats,
}

# Initialize all one-hot columns to 0
for col in all_columns:
    if col not in user_input:
        user_input[col] = 0

# Set selected categorical fields
user_input[f'brand_{brand}'] = 1
user_input[f'seller_type_{seller_type}'] = 1
user_input[f'fuel_type_{fuel_type}'] = 1
user_input[f'transmission_type_{transmission_type}'] = 1

# Convert to DataFrame
input_df = pd.DataFrame([user_input])
input_df = input_df[X_train.columns]  # Ensure column order matches

# Predict and show output
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Selling Price: ₹{prediction:,.2f}")
