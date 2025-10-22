from prophet import Prophet
import numpy as np
import pandas as pd

def convert_dataset_granularity(synthetic_arrival,granularity='1T'):
    count_df=pd.DataFrame({'ds':synthetic_arrival[:,0],'y':synthetic_arrival[:,1]})

    count_df['ds'] = pd.to_datetime(count_df['ds'], unit='s')
    count_df.set_index('ds', inplace=True)

    # Resample the DataFrame into 15-minute intervals and sum the counts
    count_df = count_df.resample(granularity).sum()
    count_df = count_df.reset_index()

    return count_df

def build_prophet_model(synthetic_arrival):
    count_df=convert_dataset_granularity(synthetic_arrival)
    m = Prophet().fit(count_df)
    return m