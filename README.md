# ai-project-2025

For my AI learning project I am looking into using AI tools to monitor and predict whether the price of BTC will rise or fall in a given time period. 

This is fairly heavy for what is meant to be a small project so in order to do this I am leveraging ChatGPT to get up and running quickly, and to help me understand some of the mathy stuff w.r.t data analysis that will need to take place.
![3pomeme](~/img/3po.jpg)

It was a bit overwhelming as for where to actually start here so I decided to focus on 1 min time deltas and make predictions by looking at previous 1 min intervals. 

# The Setup

In order to pull this off I need to pull some data from a crypto exchange. Initially I tried to use binance (which is region locked in the US), so I had to switch to Kraken which has a way cooler name anyway. Here is the python script to grab ohlcv data from Kraken: [fetch-data.py](https://github.com/dystewart/btc_predictor/blob/main/src/fetch-data.py)


This pulls data and stores it in csv format like so:
```
timestamp,open,high,low,close,volume
2025-06-30 13:08:00,107616.3,107651.4,107616.3,107651.3,0.00322576
2025-06-30 13:09:00,107670.7,107670.7,107670.7,107670.7,0.004905
2025-06-30 13:10:00,107670.7,107670.7,107670.7,107670.7,0.0
2025-06-30 13:11:00,107670.7,107670.7,107670.7,107670.7,0.000534
...
...
...
```

# Processing the Data

We have some raw data but but I still needed to figure out how I was going to train a model. I decided to use [XGBoost](https://xgboost.readthedocs.io/) as my ML tool, which helped inform me on how I might process the data I had. 

The first change I made to my data was to add indicators to the data. Here are the indicators and what they mean:
1. EMA (Exponential Moving Averages)
2. RSI (Relative Strength Index)
3. MACD (Moving Average Convergence Divergence)
4. Rolling Volatility (10-period standard deviation)

There are various strategies to employ when working with these indicators and I'm no ChatGPT helped me understand what to look for ith each and how data trends may manifest from them. 
The TLDR version of what the next code file I'll link is doing is using python library [ta](https://technical-analysis-library-in-python.readthedocs.io/en/latest/), go through our data and add indicators for each time interval. That  file is here: [build-features.py](https://github.com/dystewart/btc_predictor/blob/main/src/build-features.py)

So for those following along this is a snippet of what the new data look like:
```
timestamp,open,high,low,close,volume,rsi_14,macd,macd_signal,ema_20,ema_50,volatility
2025-06-30 13:57:00,107550.0,107550.0,107501.7,107501.7,0.20998165,25.58863343818679,-46.23506595297658,-38.81207131452796,107607.00482106133,107653.46125939385,39.33793899580966
2025-06-30 13:58:00,107507.5,107507.5,107507.5,107507.5,0.0026254,27.25331548157203,-48.35316028079251,-40.72028910778087,107597.52817143644,107647.73728843723,38.25382479052535
2025-06-30 13:59:00,107507.5,107507.5,107507.5,107507.5,0.0,27.25331548157203,-49.46160272815905,-42.46855183185651,107588.95405987106,107642.2377869299,32.28761200338686
2025-06-30 14:00:00,107533.4,107533.4,107533.4,107533.4,0.00074312,34.80661463319521,-47.70027868860052,-43.514897203205315,107583.6631970262,107637.96963842284,28.89276918371278
```

I took one more step to further make sense of the data. I made a script to determine if the price was higher (for 5 and 1 min intervals) at the close price of each row of data and add a 1 if yes, or 0 if no. Here is that file: [add-labels](https://github.com/dystewart/btc_predictor/blob/main/src/add-labels.py)

And here is the data: 
```
timestamp,open,high,low,close,volume,rsi_14,macd,macd_signal,ema_20,ema_50,volatility,target_1m,target_5m
2025-06-30 13:57:00,107550.0,107550.0,107501.7,107501.7,0.20998165,25.58863343818679,-46.23506595297658,-38.81207131452796,107607.00482106132,107653.46125939384,39.33793899580966,1,1
2025-06-30 13:58:00,107507.5,107507.5,107507.5,107507.5,0.0026254,27.25331548157203,-48.35316028079251,-40.72028910778087,107597.52817143644,107647.73728843725,38.25382479052535,0,1
2025-06-30 13:59:00,107507.5,107507.5,107507.5,107507.5,0.0,27.25331548157203,-49.46160272815905,-42.46855183185651,107588.95405987106,107642.2377869299,32.28761200338686,1,1
```

# Training the Model

The model itself is created using the XGBoost library, and works on the indicators and labels we created in our data. For my first training I have only leveraged a single label that we added previously (namely target_1m) for simplicity.

I ran the model an 80/20 split (train/test) of the data, in time order to preserve the future leaking into the past. Here is the code for running the model: [train.py](https://github.com/dystewart/btc_predictor/blob/main/src/train.py)

Running the model creates a classification report matrix which gives some good insight into what is going on under the hood. Here is a sample training run:

```
               precision    recall  f1-score   support

           0       0.73      0.84      0.78        97
           1       0.33      0.21      0.26        38

    accuracy                           0.66       135
   macro avg       0.53      0.52      0.52       135
weighted avg       0.62      0.66      0.63       135

Confusion Matrix:
 [[81 16]
 [30  8]]

```

Here is the breakdown of this output:
```
Class 0: BTC didn't go up
Precision = 0.73 → When the model predicted "no increase", it was right 73% of the time

Recall = 0.84 → It correctly identified 84% of the actual "no increase" events

F1-score = 0.78 → Harmonic mean of precision & recall (overall effectiveness)

Support = 97 → There were 97 actual "no increase" examples in the test set
--------------------------------------------------------------------------------------------------
Class 1: BTC did go up
Precision = 0.33 → When it predicted "increase", it was only right 33% of the time

Recall = 0.21 → It only caught 21% of the actual increases

F1-score = 0.26 → Weak performance for this class

Support = 38 → Fewer "increase" examples to learn from
```

In the end the model correctly classified 66% of the test data overall. But in truth this is mostly just blind luck and this model is incredibly rudimentary and lacks enough data to be very meaningful. I'm looking forward to improving the model and adding/playing around with parameters to make it much more robust. 


# End Goal

I wanted to codify a basic ML model on a set of data because I have my sights set on creating a cost/usage forecater for users of the MOC, NERC, and innabox. I think this has given me a lot of clarity in how to approach, construct and execute on a set of data and in the next quarter I'm going to be jumping into creating a model that can improve user experience across the aforementioned environments. 

