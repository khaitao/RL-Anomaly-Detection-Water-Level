import warnings
warnings.filterwarnings("ignore")
from lib import *

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class EnvTimeSeries():
    def __init__(self, code, fpath, n_steps, no_class, start, end, reward_correct, reward_correct_anomaly, reward_incorrect):
        self._code = code
        self._fpath = fpath
        #self._basin = basin
        self._ts_repo = []
        self._no_class = no_class
        self._states = 0
        self._rewards = 0
        self._check_done = 0
        self._n_steps = n_steps
        self._rwd_correct = reward_correct
        self._rwd_correct_anomaly = reward_correct_anomaly
        self._rwd_incorrect = reward_incorrect
        self._datapath = self._fpath+'/'+self._code+'.txt'

        self._df_read = pd.read_csv(self._datapath, sep='\t', parse_dates=['date'],index_col='date',usecols=["date", "wl","label"])

        if start != "":
            self._df = self._df_read.loc[start:end]

        self._df = self._df.replace(-999,np.nan)
        self._df = self._df.dropna(subset=['wl'])

        self._df['prev_wl'] = self._df['wl'].shift(1)
        self._df['diff'] = abs(self._df['wl']-self._df['prev_wl'])

        # For showing anomaly when plot the graph
        self._df['GroundTruth_Water'] = np.nan
        #self._df['GroundTruth'].loc[self._df['uea_water'] != self._df['CleanedWater']] = self._df['wl']
        self._df['GroundTruth_Water'].loc[self._df['label'] ==1] = self._df['wl']

        self._df['GroundTruth_Class'] = "Normal"
        self._df['GroundTruth_Class'].loc[self._df['GroundTruth_Water']==self._df['wl']] = "Anomaly"

        #self._df['GroundTruth_predicted'] = 0
        #self._df['GroundTruth_predicted'].loc[self._df['GroundTruth_Class']=="Anomaly"] = 1

        self._df = self._df.dropna(subset=['diff'])

        self._train_df, self._val_test_df = train_test_split(self._df, test_size=0.40, shuffle=False)
        self._val_df, self._test_df = train_test_split(self._val_test_df, test_size=0.50, shuffle=False)

        #self._num_train_df = len(self._train_df)
        self._num_anomaly = len(self._train_df.loc[self._train_df['GroundTruth_Class']=="Anomaly"])

        tf_mean = self._train_df['diff'].mean()
        tf_std = self._train_df['diff'].std()

        self._X_train_batch, self._y_train_batch = transform_data_LastVal(self._train_df['diff'].values, self._train_df["label"].values, 0, None, self._n_steps, 0, tf_mean, tf_std)
        self._X_val_batch, self._y_val_batch = transform_data_LastVal(self._val_df['diff'].values, self._val_df["label"].values, 0, None, self._n_steps, 0, tf_mean, tf_std)
        self._X_test_batch, self._y_test_batch = transform_data_LastVal(self._test_df['diff'].values, self._test_df["label"].values, 0, None, self._n_steps, 0, tf_mean, tf_std)

    # Action is Values at next time_stamp
    # state is timestamp
    def step(self, action):
        anomaly=0

        # 1 is anomaly, 0 is normal
        actual = np.argmax(self._y_train_batch[self._states])
        if (actual == 1) and (action == 1): # Corrected Anomaly
            rewards = self._rwd_correct_anomaly
        elif (actual == 1) and (action == 0): # Incorrected Anomaly
            rewards = self._rwd_incorrect
            self._check_done = self._check_done+1
            #print(f"--------- Incorrect Anomaly {self._check_done}/{self._num_anomaly} -------- ")
        elif (actual == 0 ) and (action == 0): # Correct normal
            rewards = self._rwd_correct
        else:   # Incorrected Normal
            rewards = self._rwd_incorrect

        if (self._states + 1) >= len(self._X_train_batch):
            #self._states = 0
            done= True
            self._check_done = 0
        elif (self._check_done == self._num_anomaly): # Stop training when it reach to the number of anomaly in trainig dataset
            done = True # Stop Training
            self._check_done = 0
        else:
            self._states += 1
            done = False

        return self._states, rewards, done

    def reset(self):
        self._states = 0
        self._rewards = 0
