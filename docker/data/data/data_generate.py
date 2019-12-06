#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 11:05:03 2019

@author: lizhiying
"""

from wallstreet import Stock, Call, Put
import os

from datetime import datetime

import pandas_datareader.data as web

import bs4 as bs
import pickle
import requests
import quandl 
import pandas as pd
import numpy as np
import time 


def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
    with open(r'C:\Users\HP\Desktop\stock\python for finance\sp500tickers.pickle', 'wb') as f:
        pickle.dump(tickers, f)
    return tickers

tickers = save_sp500_tickers()


tickers = [x.split()[0] for x in tickers]
tickers.remove('BRK.B')
tickers.remove('BF.B')
 

main_df = pd.DataFrame()
for i in range(len(tickers)):
    item = tickers[i]
    print(i,item)
    
    s = Stock(item)
    f = s.historical(days_back=1035,frequency = 'd')
    f.index = f['Date']
    f.drop('Date',axis = 1,inplace = True)
    
    f.columns = [x + '_'+item for x in f.columns]
    if main_df.empty:
        main_df = f
    else:
        main_df = pd.merge(main_df,f,left_index = True,right_index=True,how='outer')
            
            
main_df.to_csv('STOCK_S&P500.csv')