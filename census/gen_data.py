import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

'''
Function to generate distribution drifted data by using the census dataset from UCI repoistory

Synthetic distribution shift is introduced by splitting the data based on gender Male/Female
'''



def generate_data_drift(df,mix_n=100):
    low='<=50K'
    label_y=df['income'].apply(lambda x:0 if x==low else 1)
    label_y=label_y.values
    df['y']=label_y
    df=df.drop(['income'],axis=1)
    df_enc=pd.get_dummies(df)


    df_male=df_enc.loc[df_enc['sex_Male']==1]
    df_female=df_enc.loc[df_enc['sex_Female']==1]

    df_male_sub=df_male.sample(mix_n)
    df_male=df_male.drop(df_male_sub.index)

    df_female_sub=df_female.sample(mix_n)
    df_female=df_female.drop(df_female_sub.index)


    train_df=pd.concat([df_female,df_male_sub])
    val_df=pd.concat([df_male,df_female_sub])

    train_y=train_df['y'].values
    val_y=val_df['y'].values

    train_df=train_df.drop(['y'],axis=1)
    val_df=val_df.drop(['y'],axis=1)

    scaler=MinMaxScaler()

    train_x=scaler.fit_transform(train_df)
    val_x=scaler.transform(val_df)



    np.save("./census/data/census_train_x.npy",train_x)
    np.save("./census/data/census_val_x.npy",val_x)

    np.save("./census/data/census_train_y.npy",train_y)
    np.save("./census/data/census_val_y.npy",val_y)