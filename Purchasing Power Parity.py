#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data_path = '/Users/daoudbazaz/Downloads/log_forecasting.csv'
data = pd.read_csv(data_path)


# # Test the validity for the absolute form of Purchasing Power Parity.

# To test the validity of the absolute form of Purchasing Power Parity (PPP), we can approach this by analyzing the relationship between the exchange rate and the price levels in two countries. The absolute PPP theory suggests that the exchange rate between two countries' currencies should equal the ratio of the countries' price levels. A common way to test this is by using a linear regression model where we regress the logarithm of the exchange rate on the logarithm of the price level ratio.

# In[3]:


# Step 2: Calculate the CPI ratio as the division of 'CPI(GerHome))' by 'CPI(UK(Foreign))'
# This assumes 'CPI(GerHome))' is the domestic CPI and 'CPI(UK(Foreign))' is the foreign CPI
data['CPI Ratio'] = data['CPI(GerHome))'] / data['CPI(UK(Foreign))']


# In[4]:


# Ensure the column has been added
print("Columns in the dataset now include:", data.columns.tolist())


# In[5]:


# Step 3: Prepare the data for linear regression
# Independent variable (X) is the CPI Ratio
# Dependent variable (y) is the Nominal Exchange Rate
X = data[['CPI Ratio']].values  # Ensure it's in the correct shape for sklearn
y = data['Nominal Exchange Rate'].values


# In[6]:


# Step 4: Perform linear regression
model = LinearRegression()
model.fit(X, y)


# In[7]:


# Step 5: Extract the regression results
slope = model.coef_[0]
intercept = model.intercept_
r_squared = model.score(X, y)


# In[8]:


# Step 6: Print the results to evaluate the absolute PPP theory
print(f"Slope (Coefficient): {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared value: {r_squared}")


# TThe slope (coefficient), which we expect to be close to 1 if the absolute PPP holds.
# The intercept, which we expect to be close to 0 if the absolute PPP holds.
# The R-squared value, which indicates how well the independent variable (CPI ratio) explains the variation in the dependent variable (nominal exchange rate).
# 
# Slope (Coefficient) of -2.04: This value indicates a negative relationship between the CPI ratio and the nominal exchange rate, which is contrary to what the absolute PPP theory would predict. According to absolute PPP, we would expect a positive relationship with a coefficient close to 1, implying that changes in the CPI ratio directly correlate with proportional changes in the nominal exchange rate.
# 
# Intercept of 2.78: The non-zero intercept further deviates from the expectations of absolute PPP, which would predict an intercept close to 0 if the exchange rate directly and solely reflected the ratio of price levels between the two countries.
# 
# R-squared value of 0.34: This indicates that only about 34% of the variation in the nominal exchange rate can be explained by changes in the CPI ratio. This relatively low R-squared value suggests that other factors, beyond the scope of absolute PPP, significantly influence the nominal exchange rate.
# 
# 

# In[9]:


# Scatter plot of Nominal Exchange Rate vs. CPI Ratio
plt.figure(figsize=(10, 6))
sns.scatterplot(x='CPI Ratio', y='Nominal Exchange Rate', data=data, alpha=0.6)

# Plotting the regression line
# We'll use the model's predict method to generate y-values for the regression line based on the CPI Ratio
regression_line = model.predict(data[['CPI Ratio']].values)
plt.plot(data['CPI Ratio'], regression_line, color='red', label='Regression Line')

plt.title('Nominal Exchange Rate vs. CPI Ratio')
plt.xlabel('CPI Ratio')
plt.ylabel('Nominal Exchange Rate')
plt.legend()
plt.show()


# The negative slope of the regression line suggests an inverse relationship between the CPI Ratio and the Nominal Exchange Rate, which is contrary to what the absolute form of Purchasing Power Parity (PPP) would predict. According to the PPP theory, we would expect a positive relationship where the exchange rate increases as the ratio of the home to foreign CPI increases.
# 
# This visual evidence adds to the statistical analysis we performed earlier, suggesting that the absolute form of PPP does not hold strongly in this dataset. The considerable scatter of points also indicates that other factors may be influencing the exchange rate beyond the simple price level ratio.

# # Test the validity for the relative form of Purchasing Power Parity

# The relative form of Purchasing Power Parity (PPP) refers to the idea that the rate of change in the nominal exchange rate between two countries' currencies should equal the difference between the rate of change in their price levels.

# In[10]:


# Calculate the percentage change in the nominal exchange rate and CPIs
data['Exchange Rate Change'] = data['Nominal Exchange Rate'].pct_change()
data['Domestic CPI Change'] = data['CPI(GerHome))'].pct_change()
data['Foreign CPI Change'] = data['CPI(UK(Foreign))'].pct_change()


# In[11]:


# Calculate the difference in CPI changes
data['CPI Change Difference'] = data['Domestic CPI Change'] - data['Foreign CPI Change']


# In[12]:


# Remove the rows with NaN values that result from the pct_change method
data.dropna(inplace=True)


# In[13]:


# Make sure there are no NaN values
assert not data.isnull().values.any(), "There are NaN values in the dataframe, which should have been removed."


# In[14]:


# Make sure there are no NaN values
assert not data.isnull().values.any(), "There are NaN values in the dataframe, which should have been removed."


# In[15]:


# Prepare the independent variable (X) as the CPI Change Difference and the dependent variable (y) as the Exchange Rate Change
X = data[['CPI Change Difference']].values
y = data['Exchange Rate Change'].values


# In[16]:


# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X, y)


# In[17]:


# Extract the slope (coefficient), intercept, and R-squared value
slope = model.coef_[0]
intercept = model.intercept_
r_squared = model.score(X, y)


# In[18]:


print(f"Slope (Coefficient): {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared value: {r_squared}")


# The results you've obtained indicate the following:
# 
# Slope (Coefficient): The coefficient is approximately 0.164, which suggests a weak positive relationship between the change in exchange rates and the change in the CPI ratio. For the relative form of PPP to hold, this coefficient would ideally be closer to 1.
# Intercept: The intercept is very close to 0, suggesting that when there is no change in the CPI ratio, the change in exchange rate is negligible.
# R-squared value: An R-squared value of approximately 0.00019 is very low, indicating that the model does not explain the variability in the exchange rate changes well.
# These results suggest that the relative form of PPP does not hold strongly for the dataset, given the slope is significantly different from 1 and the low R-squared value. It's important to note that PPP is a long-term theory and may not hold in the short term due to market frictions, transaction costs, and other factors not accounted for in this basic model. Additionally, exchange rates are influenced by numerous factors beyond relative price levels, such as interest rates, political stability, and market speculati
# 

# In[19]:


# Plotting the scatter plot of Exchange Rate Change vs. CPI Change Difference
plt.figure(figsize=(12, 6))
sns.scatterplot(x='CPI Change Difference', y='Exchange Rate Change', data=data, alpha=0.6)

# Adding the regression line
# Generating predicted values for the regression line based on the independent variable
predicted_exchange_rate_change = model.predict(X)
plt.plot(data['CPI Change Difference'], predicted_exchange_rate_change, color='red', label='Regression Line')

# Finalizing the plot
plt.title('Exchange Rate Change vs. CPI Change Difference')
plt.xlabel('Difference in CPI Changes')
plt.ylabel('Percentage Change in Exchange Rate')
plt.legend()
plt.grid(True)
plt.show()


# The scatter plot you've created shows the relationship between the percentage change in the exchange rate and the difference in CPI changes. The regression line appears to be horizontal, indicating that the independent variable (the difference in CPI changes) does not explain the variability in the dependent variable (the percentage change in exchange rate).
# 
# This visualization provides a clear view that the data points are widely dispersed and do not follow a specific trend around the regression line. The nearly flat slope of the regression line corroborates the statistical finding that the coefficient is not close to 1, which would be expected if the relative PPP held.
# 
# From this plot, we can also visually confirm that the R-squared value would be low, as the regression line does not fit the data points well, implying that the model explains a very small fraction of the variance in the exchange rate changes.
# 
# This visual analysis aligns with the numerical results previously discussed, suggesting that the relative PPP does not hold for this dataset. Visualizations like this are crucial for presenting and understanding complex data relationships, providing an immediate visual context to the numerical analysis.

# In[ ]:





# In[ ]:





# In[ ]:





# # Model and forecast the real exchange rate using the Box-Jenkins modelling procedure. Estimateat least 6 models.
# 

# systematic method of identifying, fitting, and checking models for analyzing and forecasting time series data. It is primarily used for autoregressive integrated moving average (ARIMA) models. The procedure aims to identify a model that can adequately describe the time series data's statistical properties, focusing on autocorrelation, non-stationarity, and seasonality.
# 
# 

# Estimating multiple models, particularly in the context of the Box-Jenkins ARIMA methodology, involves iteratively experimenting with different combinations of autoregressive (AR), integrated (I), and moving average (MA) components to identify the best-fitting model for your time series data. Here's a structured approach to estimate at least six models for your dataset:
# 
# Step 1: Preliminary Analysis
# Before diving into model estimation, conduct a preliminary analysis to check for stationarity and seasonality in your time series data. This can involve plotting the data, examining autocorrelation and partial autocorrelation functions, and potentially applying transformations or differencing.
# 
# 
# Step 2: Model Identification
# Identify a range of potential ARIMA(p, d, q) models to estimate. Based on your preliminary analysis:
# p is the order of the AR term.
# d is the degree of differencing required to make the series stationary.
# q is the order of the MA term.
# 
# 
# Step 3: Estimation of Models
# Estimate a variety of ARIMA models with different (p, d, q) parameters. For your case, to estimate at least six models, you might start with the following combinations as an example, adjusting based on your dataset's characteristics:
# ARIMA(1,0,0): A simple autoregressive model with one lag.
# ARIMA(0,1,1): A simple moving average model with one lag on the differenced data.
# ARIMA(1,1,1): Combines both AR and MA components on the differenced data.
# ARIMA(0,1,2) or ARIMA(2,1,0): Adding complexity by increasing the order of the MA or AR terms.
# ARIMA(2,1,2): A more complex model that combines two AR and two MA terms.
# Seasonal ARIMA: If your data shows seasonality, consider estimating a seasonal ARIMA model, like SARIMA(1,1,1)(1,1,1)[S], where S is the seasonality period.
# 
# 
# Step 4: Model Fitting
# Use statistical software or programming languages like Python (with libraries like statsmodels or pmdarima) to fit these models to your data. The fitting process involves estimating the parameters of each model to best fit the historical data.
# 
# 
# Step 5: Model Comparison and Selection
# After fitting the models, compare them based on criteria like the Akaike Information Criterion (AIC), Bayesian Information Criterion (BIC), or the Mean Squared Error (MSE) of forecasts. Lower values of AIC or BIC indicate a better-fitting model, considering both the goodness of fit and the complexity of the model.
# 
# 
# Step 6: Diagnostic Checks
# For the best-performing models, conduct diagnostic checks to assess the adequacy of the model fit. This involves examining the residuals to ensure they resemble white noise (no autocorrelation).
# 
# Would you like to proceed with estimating these models on your dataset using Python? If so, we can start by conducting a preliminary analysis to identify the order of differencing required to make your series stationary.

# # Step 1: Preliminary Analysis

# # Plotting the Time Series

# In[20]:


data['Real Exchange Rate'].plot(title='Real Exchange Rate Over Time')
plt.xlabel('Time')
plt.ylabel('Real Exchange Rate')
plt.show()


# # Performing a Stationarity Test

# In[21]:


from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(data['Real Exchange Rate'])
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')


# # Analyzing Autocorrelation and Partial Autocorrelation

# In[22]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(data['Real Exchange Rate'], lags=20)
plot_pacf(data['Real Exchange Rate'], lags=20)
plt.show()


# ADF Test:
# 
# ADF Statistic: -2.4019
# p-value: 0.1412
# The ADF statistic and the p-value above 0.05 suggest that we cannot reject the null hypothesis; the series is not stationary. Typically, a non-stationary series would require differencing to achieve stationarity before ARIMA modeling can be appropriately applied.
# 
# Time Series Plot:
# The plot shows fluctuations over time with no clear or consistent trend. There are ups and downs, but no clear pattern of seasonality or trend that strongly stands out. This may imply the need for a differencing to stabilize the mean, although the lack of trend suggests that extensive differencing may not be necessary.
# 
# Autocorrelation Plot:
# The ACF plot shows a gradual decline in correlation as the lags increase, which is indicative of a non-stationary time series. This reinforces the ADF test result and suggests that the series would benefit from differencing.

# # transformations or differencing

# In[23]:


# Step 1: Difference the data to achieve stationarity
data['Differenced_Real_Exchange_Rate'] = data['Real Exchange Rate'].diff().dropna()


# In[24]:


# Check stationarity again
adf_result_diff = adfuller(data['Differenced_Real_Exchange_Rate'].dropna())
print(f'ADF Statistic (Differenced Data): {adf_result_diff[0]}')
print(f'p-value (Differenced Data): {adf_result_diff[1]}')


# In[25]:


# Plot ACF and PACF for the differenced series
plot_acf(data['Differenced_Real_Exchange_Rate'].dropna(), lags=20, title='ACF for Differenced Series')
plot_pacf(data['Differenced_Real_Exchange_Rate'].dropna(), lags=20, title='PACF for Differenced Series')
plt.show()


# The results from the differenced data are very clear:
# 
# ADF Test on Differenced Data:
# 
# ADF Statistic: -53.5746
# p-value: 0.0
# The ADF test statistic is far below any typical critical value, and the p-value is 0.0, which strongly suggests that the differenced data is stationary. You do not need to difference the data again.
# 
# ACF for Differenced Series:
# The ACF plot shows a significant spike at lag 0 and then quickly decays towards zero, which is typical for a differenced stationary series.
# 
# PACF for Differenced Series:
# The PACF plot shows a significant spike at lag 1 and then cuts off, which suggests an AR(1) component might be appropriate for the data.
# 
# Based on these plots, an ARIMA model with d=1 (since we've differenced the data once) seems appropriate. The PACF plot suggests p could be 1 since there's a significant spike at lag 1 and the rest are insignificantly different from zero. The ACF plot decays quickly, which could imply a small q value as well.
# 
# With these insights, we can refine the set of ARIMA models to estimate. We should at least estimate models within the vicinity of ARIMA(1,1,1), possibly including:
# 
# ARIMA(1,1,0)
# ARIMA(1,1,1)
# ARIMA(1,1,2)
# ARIMA(0,1,1)
# ARIMA(2,1,1)
# ARIMA(2,1,0

# # Step 3: Estimation of Models
# We estimate six ARIMA models with different configurations to capture various dynamics in the time series.

# In[29]:


import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt


# In[42]:


# Define a range of models to estimate
model_parameters = [(p, 1, q) for p in range(3) for q in range(3)]

# Step 3: Estimating Models
aic_values = {}
for param in model_parameters:
    model = ARIMA(data['Real Exchange Rate'], order=param)
    results = model.fit()
    aic_values[param] = results.aic
    print(f'ARIMA{param} - AIC:{results.aic}')
          
       


# In[43]:


# Step 4: Model Selection
best_model_param = min(aic_values, key=aic_values.get)
best_model = ARIMA(data['Real Exchange Rate'], order=best_model_param)
best_model_fit = best_model.fit()


# In[44]:


# Print the best model summary
print(f'Best Model: ARIMA{best_model_param}')
print(best_model_fit.summary())


# In[34]:


# Step 5: Diagnostic Checking
best_model_fit.plot_diagnostics(figsize=(15, 12))
plt.show()


# Standardized Residuals Plot:
# This plot shows the residuals from your model over time. What we hope to see is that there's no pattern in the residuals—they should look like "white noise". There appears to be an outlier, which is quite large compared to the others. This might indicate an unusual event at that particular time period.
# 
# Histogram plus Estimated Density Plot:
# The histogram shows the distribution of the residuals along with a Kernel Density Estimate (KDE) and the Normal distribution for comparison. The KDE gives an indication of the estimated distribution of the residuals. The closer this is to the Normal distribution, the better, as it suggests that the residuals are normally distributed—a common assumption in time series modeling.
# 
# Normal Q-Q Plot:
# This is a quantile-quantile plot which compares the distribution of the residuals with a normal distribution. If the points fall along the red line, then the residuals are normally distributed. It looks like the residuals may have heavy tails, as indicated by the deviation from the line in the tails.
# 
# Correlogram:
# This is a plot of the autocorrelation function of the residuals. Ideally, we want to see that there's no autocorrelation in the residuals, which would be indicated by all bars falling within the blue shaded area (representing a 95% confidence interval). The plot suggests that the residuals are not autocorrelated, which is good.
# 
# Based on these diagnostics, your model seems to capture the structure of the time series well, except for the potential outlier. You might want to investigate the outlier to see if it was due to a special circumstance or if it might be a data error. If it's a one-time event that won't repeat in the future, it might not affect your forecasts significantly. However, if it's an error or indicative of a deeper issue in the data, you should address it.
# 
# Before making forecasts, it's also important to ensure that the underlying assumptions of your model are met. Given the heavy tails indicated in the Q-Q plot, you might want to consider models that can handle non-normal residuals or apply some transformation to the data. If you're satisfied with the diagnostic checks, you can proceed with using the model for forecasting.

# In[47]:


# Step 6: Forecasting
forecast_steps = 5
forecast = best_model_fit.get_forecast(steps=forecast_steps)
forecast_index = data.index[-1] + pd.Index(range(1, forecast_steps+1))
forecast_conf_int = forecast.conf_int()

# Plot the historical data and forecasts
plt.figure(figsize=(10, 6))
plt.plot(data['Real Exchange Rate'], label='Historical Real Exchange Rate')
plt.plot(forecast_index, forecast.predicted_mean, label='Forecast', color='red')
plt.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Forecast of Real Exchange Rate')
plt.xlabel('Time Period')
plt.ylabel('Real Exchange Rate')
plt.legend()
plt.show()

# Output the forecasted values
print(f"Forecasted Real Exchange Rate for the next {forecast_steps} periods:")
print(forecast.predicted_mean)


# In[ ]:




