import sys
# Add venv site-packages if available
sys.path.insert(0, r'c:\Users\hp\HousePricePrediction\venv\Lib\site-packages')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import pickle

# Load locations
loc_df = pd.read_csv('location.csv')
locations = loc_df['Location'].unique().tolist()

# Improved synthetic data generation with realistic per-sqft rates (INR)
rng = np.random.default_rng(42)
samples = []
for loc in locations:
    # per-sqft rate depends on location; sample between 2000 and 10000 INR/sqft
    rate = rng.uniform(2000, 10000)
    for _ in range(150):
        area = float(rng.integers(300, 4000))
        bedrooms = int(rng.integers(1, 5))
        resale = int(rng.choice([0,1], p=[0.7,0.3]))
        # price (INR) = area * rate_per_sqft + bedroom premium + resale adjustment + noise
        bedroom_premium = bedrooms * rng.uniform(100000, 400000)
        resale_adj = -resale * rng.uniform(20000, 150000)
        noise = rng.normal(0, 500000)
        price = area * rate + bedroom_premium + resale_adj + noise
        # clip to sensible minimum
        price = max(price, 500000)
        samples.append({'Location':loc, 'Area':area, 'No_of_Bedrooms':bedrooms, 'Resale':resale, 'Price':round(price,2)})

df = pd.DataFrame(samples)

# Features / target
X = df[['Location','Area','No_of_Bedrooms','Resale']]
y = df['Price']

# Build pipeline
column_trans = make_column_transformer((OneHotEncoder(sparse=False, handle_unknown='ignore'), ['Location']), remainder='passthrough')
scaler = StandardScaler()
model = RandomForestRegressor(n_estimators=100, random_state=42)
pipe = make_pipeline(column_trans, scaler, model)

print('Training on synthetic data, samples=', len(X))
pipe.fit(X, y)

# Save model
pickle.dump(pipe, open('Nofeature.pkl', 'wb'))
print('Saved Nofeature.pkl')
