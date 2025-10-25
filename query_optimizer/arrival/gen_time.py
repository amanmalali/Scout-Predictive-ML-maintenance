import numpy as np
import pandas as pd
#TODO EXPLAIN FILE
def lambda_t(time,data_ts,data_counts):
    return np.interp(time,data_ts,data_counts,0,0)


def simulate_points(num_points,data_ts,data_counts):
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



def add_timestamps_simple(sim_x,sim_y,ts,counts,future_hours=48,historical_hours=None):
    last_ts=ts[-1]
    search_ts=last_ts-future_hours*60*60
    idx=find_nearest(ts,search_ts)
    events=simulate_points(len(sim_x),ts[idx:],counts[idx:])
    cols=[]
    for f in range(sim_x.shape[1]):
        cols.append('feat_'+str(f))
    # cols.append("y")
    # all_data=np.concatenate([train_data,val_data],axis=0)
    df=pd.DataFrame(data=sim_x,columns=cols)
    df['timestamp']=events
    df['y']=sim_y

    events_per_sec=(len(sim_x))/(ts[-1]-ts[idx])

    historical_events=int((ts[idx]-ts[0])*events_per_sec)

    events=simulate_points(historical_events,ts[:idx],counts[:idx])

    return df,events


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


    






# ts=np.load('./data/timestamps.npy')
# counts=np.load('./data/counts.npy')

# # plt.scatter(ts,counts)

# # plt.show()

# val_data=np.load('./data/concept_val_x.npy')
# train_data=np.load('./data/concept_train_x.npy')

# df,events=add_timestamps_simple(train_data,val_data,ts,counts)
# print(df.head())
# last_ts=ts[-1]
# search_ts=last_ts-48*60*60
# idx=find_nearest(ts,search_ts)
# print(ts[idx])
# events=simulate_points(len(val_data)+len(train_data),ts[idx:],counts[idx:])

# event,count=np.unique(events,return_counts=True)
# plt.scatter(event,count,s=1)

# plt.show()


# events_per_sec=(len(val_data)+len(train_data))/(ts[-1]-ts[idx])

# historical_events=int((ts[idx]-ts[0])*events_per_sec)

# events=simulate_points(historical_events,ts[:idx],counts[:idx])

# event,count=np.unique(events,return_counts=True)
# plt.scatter(event,count,s=1)

# plt.show()
