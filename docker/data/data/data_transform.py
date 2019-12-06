#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 20:26:03 2019

@author: lizhiying
"""

import pandas as pd 
import numpy as np
import talib
data = pd.read_csv('/Users/lizhiying/Desktop/Reinforcement/stock_trading_rl/STOCK_S&P500.csv')

data_np = np.array(data)

np_nodate = data_np[:,1:]


OPEN = []
HIGH = []
LOW = []
CLOSE = []
ADJ_CLOSE = []
VOLUME = []
CODE = []

for i in range(500):
    OPEN.append(np_nodate[:,6*i])
    HIGH.append(np_nodate[:,6*i+1])
    LOW.append(np_nodate[:,6*i+2])
    CLOSE.append(np_nodate[:,6*i+3])
    ADJ_CLOSE.append(np_nodate[:,6*i+4])
    VOLUME.append(np_nodate[:,6*i+5])
    
    CODE.append(data.columns[1:][6*i])
    
    
OPEN = np.array(OPEN).reshape(-1)
HIGH = np.array(HIGH).reshape(-1)
LOW = np.array(LOW).reshape(-1)
CLOSE = np.array(CLOSE).reshape(-1)
ADJ_CLOSE = np.array(ADJ_CLOSE).reshape(-1)
VOLUME = np.array(VOLUME).reshape(-1)
DATE = data_np[:,0]

DATE = np.tile(DATE,500)



CODE = [item[5:] for item in CODE]
CODE = np.repeat(CODE,714)

PERCENT = CLOSE-OPEN/OPEN


VOLUME  = VOLUME/1000000
VOLUME = [str(round(item,2))+'M' for item in VOLUME]


PERCENT = np.diff(CLOSE)/CLOSE[:-1]

PERCENT = np.insert(PERCENT,0,0)
for i in range(0,500):
    PERCENT[i*714] = 0

price = np.c_[DATE,OPEN,HIGH,LOW,VOLUME,PERCENT,CODE]