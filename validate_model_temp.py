import pickle
import pandas as pd
from pathlib import Path
p = Path('Nofeature.pkl')
print('exists', p.exists())
with p.open('rb') as f:
    pipe = pickle.load(f)
print('type', type(pipe))
X = pd.DataFrame([['Koramangala', 1200.0, 3, 0]], columns=['Location', 'Area', 'No_of_Bedrooms', 'Resale'])
print('predict', pipe.predict(X))
