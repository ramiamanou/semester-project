import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('data_project.csv')

df.head()

df.info()

df.isna().sum()

df = df[df['yr_renovated'] <= 0]
df.info()

df = df.drop(columns=['yr_renovated'])
df.head()

df = df[df['area'] <= 60000]
df.info()

df = df[df['price'] <= 30466140000]
df.info()

df.describe()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='area', y='price', data=df)
plt.title('Scatter Plot of Area vs. Price')
plt.xlabel('Area (sq. meter.)')
plt.ylabel('Price')
plt.show()

plt.figure(figsize=(10, 6))
sns.regplot(x='area', y='price', data=df)
plt.title('Regression Line Plot of Area vs. Price')
plt.xlabel('Area (sq. ft.)')
plt.ylabel('Price')
plt.show()

sns.pairplot(df, x_vars=['area', 'bath', 'bed', 'age'], y_vars='price', height=5, aspect=0.7)
plt.show()

df.describe()

df['log_area'] = np.log(df['area'] + 1)
df['log_price'] = np.log(df['price'] + 1)

plt.figure(figsize=(10, 6))
sns.regplot(x='log_area', y='log_price', data=df)
plt.title('Regression Line Plot of Area vs. Price')
plt.xlabel('Area (sq. meter.)')
plt.ylabel('Price')
plt.show()

features = ['bed', 'bath', 'age' , 'log_area', 'view', 'condition'] 
X = df[features]
y = df['log_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')