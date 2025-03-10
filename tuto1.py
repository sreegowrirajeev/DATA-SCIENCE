import pandas as pd
import statsmodels.api as sm

# Load the dataset from the specified file path
file_path = "Advertising.csv"
df = pd.read_csv(file_path)

# Remove the unnecessary index column
df = df.drop(columns=["Unnamed: 0"])

# Define the predictor variables (TV, radio, newspaper) and response variable (sales)
X = df[['TV', 'radio', 'newspaper']]
y = df['sales']

# Add a constant for the intercept term to account for the bias term in the regression model
X = sm.add_constant(X)

# Fit the multiple linear regression model using Ordinary Least Squares (OLS)
model = sm.OLS(y, X).fit()

# Extract key statistics
rse = model.mse_resid ** 0.5  # Compute the Residual Standard Error (RSE)
r_squared = model.rsquared    # Compute the R-squared value to measure model fit
f_statistic = model.fvalue    # Compute the F-statistic to test overall model significance

# Print the results
print("Residual Standard Error (RSE):", rse)  # Print RSE
print("R-squared (R²):", r_squared)  # Print R-squared value
print("F-statistic:", f_statistic)  # Print F-statistic
print(model.summary())  # Print the full regression summary
