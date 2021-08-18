from os import write
import pystan
import streamlit as st
import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics, cross_validation
from fbprophet.plot import plot_cross_validation_metric
import base64

from streamlit import errors

st.set_page_config(page_title="FB Prophet-Example",
                   page_icon="📈", initial_sidebar_state="expanded")

st.title("Automated Time Series Forecasting")


"""
This data app uses Facebook's open-source Prophet library to automatically generate future forecast values from an imported dataset.
You'll be able to import your data from a CSV file, visualize trends and features, analyze forecast performance, and finally download the created forecast 😵 

**In beta mode**

Created by Zach Renwick: https://twitter.com/zachrenwick

Re-created and enhanced: Indrashis Paul
"""

"""
### Step 1: Import Data

Only two columns are applicable. The input to Prophet is always a dataframe with two columns. The (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The target column must be numeric, and represents the measurement we wish to forecast.

"""

df = st.file_uploader(
    'Upload the time series csv file here.', type='csv')

if df is not None:
    data = pd.read_csv(df)
    data['ds'] = pd.DataFrame(data.iloc[:, 0])

    st.write(data)

    max_date = data['ds'].max()


"""
### Step 2: Select Forecast Horizon

Note: Forecasts become less accurate with larger forecast horizons.
"""

periods_input = st.number_input(
    'How many periods would you like to forecast into the future?', min_value=1, max_value=365)

if df is not None:
    m = Prophet()
    m.fit(data)


"""
### Step 3: Visualize Forecast Data

The below visual shows future predicted values. "yhat" is the predicted value, and the upper and lower limits are (by default) 80% confidence intervals.
"""

if df is not None:
    future = m.make_future_dataframe(periods=periods_input)

    forecast = m.predict(future)
    fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    fcst_filtered = fcst[fcst['ds'] > max_date]
    st.write(fcst_filtered)

    """
    The next visual shows the actual (black dots) and predicted (blue line) values over time.
    """
    fig1 = m.plot(forecast)
    st.write(fig1)

    """
    The next few visuals show a high level trend of predicted values, day of week trends, and yearly trends (if dataset covers multiple years). The blue shaded area represents upper and lower confidence intervals.
    """
    fig2 = m.plot_components(forecast)
    st.write(fig2)


"""
### Step 4: Download the Forecast Data

The below link allows you to download the newly created forecast to your computer for further analysis and use.
"""

if df is not None:
    csv_exp = fcst_filtered.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(csv_exp.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
    st.markdown(href, unsafe_allow_html=True)
