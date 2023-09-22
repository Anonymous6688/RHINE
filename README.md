# RHINE: A Regime-Switching Model with Nonlinear Representation for Discovering and Forecasting Regimes in Financial Markets

A <ins>R</ins>egime-switc<ins>HI</ins>ng model with <ins>N</ins>onlinear r<ins>E</ins>presentation which is capable of discovering and exploiting the significant underlying patterns in time series.

## Motivation
![Image text](https://github.com/Anonymous6688/RHINE/raw/master/images/Regime1.png)

![Image text](https://github.com/Anonymous6688/RHINE/raw/master/images/trends.png)

## Framework of RHINE

![Image text](https://github.com/Anonymous6688/RHINE/raw/master/images/framework.png)

## Environment
The code was written in Python 3.9, and has the following dependencies:
arch==5.3.1
beautifulsoup4==4.11.1
cvxpy==1.2.1
graphviz==0.20
joblib==1.1.0
matplotlib==3.5.1
more_itertools==8.12.0
munkres==1.1.4
networkx==2.8.1
numpy==1.21.6
pandas==1.4.2
PyPDF2==2.9.0
pytrends==4.8.0
requests==2.27.1
scikit_learn==1.1.1
scipy==1.8.0
seaborn==0.11.2
statsmodels==0.13.2
sympy==1.10.1
tqdm==4.63.0
yfinance==0.1.72

## Download Dataset：

### *Stock1*
The first dataset *Stock1*, with 503 stocks collected from the S\&P 500, composed of daily OHCLV (open, high, close, low, volume) data from 2012-01-04 to 2022-06-22. [https://ca.finance.yahoo.com/](https://ca.finance.yahoo.com/)       
### *Stock2*
The second *Stock2*, composed of intra-day market hours OHCLV data from 2017-05-16 to 2017-12-06, for 467 stocks from    [https://www.kaggle.com/datasets/borismarjanovic/daily-and-intraday-stock-price-data](https://www.kaggle.com/datasets/borismarjanovic/daily-and-intraday-stock-price-data).  

 ### Volatility estimator

By Appling the classical volatility estimator, we can obtain the volatility time series of stock datasets. The volatility estimator is defined as: 

$V_t = \sqrt{\sum_{t=1}^n (r_t)^2}$, where $r_t=\ln (c_t/c_{t-1})$ and $c_t$ is the closing price at time $t$.

## Run

run Syd_main.py on Synthetic dataset. 

run Stock1_main.py on Stock1 dataset. 

run Stock2_main.py on Stock2 dataset (real-time forecasting). 

## Result (the complete experimental results are shown in the paper and appendix)
### Result on Syd time series (Regime-identification)
![Image text](https://github.com/Anonymous6688/RHINE/raw/master/images/Syd.png)

### Result on Stock1 (Regime-identification and switching)
![Image text](https://github.com/Anonymous6688/RHINE/raw/master/images/Stock1.png)
### Result on Stock2 (Real-time forecasting)
![Image text](https://github.com/Anonymous6688/RHINE/raw/master/images/Stock2.png)
### N-beats Model (SOTA DL model) Result on Stock2 (Real-time forecasting)
![Image text](https://github.com/Anonymous6688/RHINE/raw/master/images/Nbeats.png)
### Models' forecasting performance, in terms of RMSE
![Image text](https://github.com/Anonymous6688/RHINE/raw/master/images/Table.png)
### Computation time on the three data sets (Linear-log scale)
![Image text](https://github.com/Anonymous6688/RHINE/raw/master/images/Time.png)
