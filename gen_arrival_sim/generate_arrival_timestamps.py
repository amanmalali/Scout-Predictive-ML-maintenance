import numpy as np
import pandas as pd

def lambda_t(time,data_ts,data_counts):
    return np.interp(time,data_ts,data_counts,0,0)



def simulate_points(num_points,data_ts,data_counts):
    """
    Simulate `num_points` event arrival timestamps using 
    non-homogeneous Poisson process and thinning.

    Inputs
    ------
    num_points : int
        Number of event times to generate.
    data_ts : array-like
        Timestamps defining the simulation window [min(data_ts), max(data_ts)].
    data_counts : array-like
        Values used to define the time-varying intensity λ(t) (via `lambda_t`).

    Output
    ------
    events : numpy.ndarray
        Sorted array of length `num_points` containing simulated event timestamps.
    """
    lambda_max=max(data_counts)

    events=[]
    while len(events)<num_points:
        random_times=np.random.randint(min(data_ts),max(data_ts),num_points)

        intensities=lambda_t(random_times,data_ts,data_counts)
        u_val=np.random.uniform(0,1,num_points)

        thinning_condition=u_val<=intensities/lambda_max

        events.extend(random_times[thinning_condition])
    
    events=np.sort(np.random.choice(events,num_points,replace=False))

    return events

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx



def add_timestamps_to_data(sim_x,sim_y,ts,counts,future_hours=48,tabular=True):
    """
    Assign timestamps to synthetic samples by treating the last `future_hours` as the “future” window,
    and simulate the remaining earlier period as “historical” arrivals.

    Returns a DataFrame for the future window where each synthetic sample is paired with a simulated
    arrival timestamp and its label, plus a separate array of simulated historical arrival timestamps.
    The number of historical arrivals is chosen to match the average arrival rate implied by the future
    window (i.e., arrivals per second in the future window are extrapolated to the earlier period).

    Parameters
    ----------
    sim_x : array-like
        Synthetic features (N, D) if `tabular=True`, otherwise only length N is used.
    sim_y : array-like
        Labels/targets of length N.
    timestamps : array-like
        Monotone timeline used to define the overall period and the split into future vs historical.
    arrival_counts : array-like
        Values aligned with `timestamps` used to drive the arrival intensity.
    future_hours : int, default=48
        Size of the future window (hours) at the end of the timeline.
    tabular : bool, default=True
        If True, include `feat_0..feat_{D-1}` columns in the returned DataFrame.

    Returns
    -------
    future_df : pandas.DataFrame
        N-row DataFrame for the future window with columns: optional `feat_*`, plus `timestamp` and `y`.
    historical_timestamps : numpy.ndarray
        Simulated arrival timestamps for the earlier (historical) period.
    """
    last_ts=ts[-1]
    search_ts=last_ts-future_hours*60*60
    idx=find_nearest(ts,search_ts)
    events=simulate_points(len(sim_x),ts[idx:],counts[idx:])
    cols=[]
    if tabular:
        for f in range(sim_x.shape[1]):
            cols.append('feat_'+str(f))
        df=pd.DataFrame(data=sim_x,columns=cols)
    else:
        df=pd.DataFrame()
    
    df['timestamp']=events
    df['y']=sim_y

    events_per_sec=(len(sim_x))/(ts[-1]-ts[idx])

    historical_events=int((ts[idx]-ts[0])*events_per_sec)

    history_events_with_ts=simulate_points(historical_events,ts[:idx],counts[:idx])

    return df, history_events_with_ts


"""
LEGACY: combined into add_timestamps_to_data
def add_timestamps_images(sim_x,sim_y,ts,counts,future_hours=48,historical_hours=None):
    last_ts=ts[-1]
    search_ts=last_ts-future_hours*60*60
    idx=find_nearest(ts,search_ts)
    events=simulate_points(len(sim_x),ts[idx:],counts[idx:])
    cols=[]

    cols.append("y")
    df=pd.DataFrame(data=sim_y,columns=cols)
    df['timestamp']=events

    events_per_sec=(len(sim_x))/(ts[-1]-ts[idx])

    historical_events=int((ts[idx]-ts[0])*events_per_sec)

    events=simulate_points(historical_events,ts[:idx],counts[:idx])

    return df,events
"""