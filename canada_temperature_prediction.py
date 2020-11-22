import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

def read_and_prepare_dataframe(start_date='1980-01-01'):
    """Function will read the dataset and prepare the dataframe for the forecast model.
    Dataframe will only contain data for Canada beginning at start_date."""
    
    # Read the dataset and rename 'dt' to 'Date'
    df = pd.read_csv('Data/GlobalLandTemperaturesByCountry.csv', parse_dates=['dt'])
    df.rename(columns={'dt':'Date'}, inplace=True)
    
    # Filter for Canada
    df = df[df['Country']=='Canada']
    
    # Filter out data prior to start date
    df = df[df['Date'] >= start_date]
    
    # To ensure data is sorted
    df = df.sort_values('Date')
    
    # Set index to Date and return the final dataframe
    return df.set_index('Date')

def time_series_train_test_split_y(df, test_date_start):
    """Splits the time series dataframe, df, into train and test dataframes 
    for y only at the specified start date for testing, test_date_start.
    The function returns all of y, y_train, y_test.
    """
    test_date_cutoff = datetime.strptime(test_date_start, '%Y-%m-%d')
    train_date_cutoff = test_date_cutoff - timedelta(1)
    
    y = df['AverageTemperature']
    y_train = y.loc[:train_date_cutoff]
    y_test = y.loc[test_date_cutoff:]
    
    return y, y_train, y_test

def optimize_sarima_parameters(y, **kwargs):
    """Returns optimized (p,d,q),(P,D,Q,m) parameters based on specified test and metric.
    Optimization will be done on the full dataset (train and test).
    It is assumed that d=1, D=1, m=12, and seasonal=True."""
    
    opt = pm.auto_arima(y, d=1, D=1, m=12, seasonal=True, **kwargs)
    return opt.get_params()['order'], opt.get_params()['seasonal_order']


def train_sarima_model(y_train, order, seasonal_order, plot_diagnostics=True, **kwargs):
    """Trains a SARIMAX model based on the training data, y_train, 
    SARIMA parameters, order and seasonal_order, and other keywords.
    Will also return a diagnostics plot of the model if plot_diagnostics=True."""
    
    # Fit Model
    model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order, **kwargs)
    results = model.fit()

    # Plot diagnostics, if true
    if plot_diagnostics:
        results.plot_diagnostics(figsize=(15,8))
        plt.show()
    
    return results

def test_sarima_model(y, y_test, results, **kwargs):
    """Use the model, results, to make prediction on the test dataset, y_test.
    Prints out the root mean squared error, the R^2 value and a graph of 
    observed vs. predicted average monthly temperature for these predictions."""
    
    # Get predictions
    pred = results.get_prediction(start=y_test.index.min(), end=y_test.index.max(), **kwargs)
    y_pred = pred.predicted_mean
    pred_ci = pred.conf_int()

    # Calculate some metrics and print them out
    rmse = ((y_pred - y_test) ** 2).mean() ** 0.5
    print('Root Mean Squared Error =', rmse)
    
    r2 = r2_score(y_pred, y_test)
    print('R^2 =', r2)
    
    # Graph
    ax = y.plot(label='observed')
    y_pred.plot(ax=ax, label='predicted', alpha=.7, figsize=(15, 8))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    plt.title('Average Monthly Temperature: Observed vs. Predicted')
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature')
    plt.legend()
    plt.show()
    
def forecast_temperature(y, results, end_date, **kwargs):
    """Uses the SARIMAX model, results, to forecast the average monthly temperature
    from the end of the dataset, y, until the specified end date, end_date.
    Returns a table containing predicted mean, standard error, and the lower and upper 
    95% confidence interval bounds.
    Function will also print out a graph containing the forecasted average monthly temperature."""
    
    # Forecast
    forecast = results.get_prediction(start=(y.index.max() + timedelta(1)),
                                      end=end_date,
                                      **kwargs)
    y_forecast = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    # Graph forecast
    ax = y.plot(label='observed')
    y_forecast.plot(ax=ax, label='forecasted', alpha=.7, figsize=(15, 8))
    ax.fill_between(forecast_ci.index,
                    forecast_ci.iloc[:, 0],
                    forecast_ci.iloc[:, 1], color='k', alpha=.2)
    plt.title('Forecasted Average Monthly Temperature')
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature')
    plt.legend()
    plt.show()
    
    # Generate forecast table
    forecast_table = forecast.summary_frame()
    forecast_table.index.rename('Date',inplace=True)
    forecast_table = round(forecast_table, 2)
    
    return forecast_table
    

def run_forecast(data_start_date='1980-01-01', forecast_end_date='2015-12-01',
                 test_date_start='2010-01-01', test_model=True):
    """Function will call all the necessary functions to run the forecast and return the final table.
    Can specify the start date for the dataset used to train the forecast, the end date for the forecast,
    and the cutoff date for splitting the test data.
    If desired, the function can output the prediciton results from the test data.
    """
    
    # 1. Prepare dataframe.
    df = read_and_prepare_dataframe(start_date=data_start_date)
    
    # 2. Split dataframe into train and test.
    y, y_train, y_test = time_series_train_test_split_y(df=df, test_date_start=test_date_start)
    
    # 3. Optimize parameters.
    opt_order, opt_seasonal_order = optimize_sarima_parameters(y,
                                                               trend='c', start_p=0, start_q=0, test='adf', 
                                                               max_order=6, stepwise=True, trace=False)
    
    # 4. Train SARIMAX model.
    results = train_sarima_model(y_train=y_train, order=opt_order, seasonal_order=opt_seasonal_order, 
                                 plot_diagnostics=False,
                                 trend='c', enforce_stationarity=False, enforce_invertibility=False)
    
    # 5. Test the model (optional).
    if test_model:
        test_sarima_model(y=y, y_test=y_test, results=results, dynamic=True)
    
    # 6. Make the forecast with the model and return the table containing the results.
    table = forecast_temperature(y=y, results=results, end_date=forecast_end_date)
    return table

if __name__ == '__main__':
    forecast_table = run_forecast()
    display(forecast_table)
