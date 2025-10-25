import pandas as pd
import numpy as np

# column_names=['job_name','inst_id','user','status','start_time','end_time']
column_names=['job_name','task_name','inst_num','status','start_time','end_time','plan_cpu','plan_mem','plan_gpu','gpu_type']
df=pd.read_csv("./data/pai_task_table.csv",names=column_names)
# df=df.loc[df['status']=='Running']
df['start_time']=df.start_time.apply(pd.Timestamp, unit='s', tz='Asia/Shanghai')
# df['start_time']=pd.to_datetime(df['start_time'])
df=df.sort_values(by=['start_time'])
df=df.set_index('start_time')
# print(df.head())
new_df=df.resample('1min').count()
new_df=new_df.reset_index()
new_df=new_df.loc[new_df['job_name']>0]
new_df['start_time']=new_df['start_time'].apply(lambda ts: ts.replace(month=1))
new_df['timestamp']=(new_df['start_time'].astype(np.int64) // 10 ** 9)+28800
new_df=new_df.drop(['start_time'],axis=1)
new_df=new_df.sort_values(by=['timestamp'])
new_df=new_df.groupby(by=['timestamp']).sum()
new_df=new_df.reset_index()

# print(new_df.head())
ts=new_df['timestamp'].values
counts=new_df['job_name'].values

np.save("./data/timestamps.npy",ts)
np.save("./data/counts.npy",counts)

