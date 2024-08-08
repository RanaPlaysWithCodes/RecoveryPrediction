import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
dataset = pd.read_csv('training_data.csv')

dataset.dropna(inplace=True)

y = dataset['recovery']
x = dataset.drop('recovery',axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
y_model_test_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score

print(f'Linear Regression R2 Score: {r2_score(y_test, y_model_test_pred)}')


import pickle
pickle.dump(model, open('lrg_model.pkl','wb'))

