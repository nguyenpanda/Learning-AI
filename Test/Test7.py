
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
#
# # Read data from CSV file into a DataFrame
# df = pd.read_csv('TestData.csv', header=None, names=['Target', 'Feature3', 'Feature2'])
#
# # Feature selection
# X = df[['Feature2', 'Feature3']]
# y = df['Target']
#
# # Create and train the linear regression model using all data
# model = LinearRegression()
# model.fit(X, y)
#
# # Predict for new data
# new_data = pd.DataFrame({'Feature2': [20.5], 'Feature3': [7]})
# predicted_value = model.predict(new_data)
#
# print(f"Predicted value for new data: {predicted_value[0]}")
#
# print(model.coef_)
# print(model.intercept_)
