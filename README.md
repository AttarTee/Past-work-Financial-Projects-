# Past-work-Financial-Projects-

main report is in TSFA Project 2022.ipynb


# TAX-FREE SAVINGS ACCOUNT Project

by Alexandros Taderera

### Executive Summary:
> South Africa's Tax-Free Savings Account (TFSA) investors are an underserved demographic with macroeconomic significance. A large portion of TFSA investors are part-time investors. Due to the nature of part-time investors, the greatest marginal contribution comes from implementing basic risk-management protocols.

> The South African Treasury released a white paper titled [***"INCENTIVISING NON-RETIREMENT SAVINGS"***](http://www.treasury.gov.za/comm_media/press/2012/Incentivising%20non-retirement%20savings.pdf) *[2012]*. This paper outlines the financial vulnerability of the average South African household and why we need rapid and immediate policy action.
> From 2016-17 the [***"A STUDY OF TAX-FREE SAVINGS ACCOUNT TAKEUP IN SOUTH AFRICA"***](https://www.intellidex.co.za/wp-content/uploads/2017/07/TFSA-Survey-Report-2017-Live-Version-FINAL.pdf) reports were published following the introduction of ***collective investment schemes*** as part of Tax-Free Savings. The uptake has been notable.


|Year |Accounts Opened|Amount Held |1st-time Buyers|
|:-----|:----:|----:|----:|
|2015 |35 384 |**R284m** |32%|
|2016 |262 493 |**R2 600m** | 21% |
|2017 |207 172|**R5 174m**| --|

###### MODELLING PERSCPECTIVE: 
>  I use modelling techniques of [*"A Semiparametric Graphical Modelling Approach For Large-Scale Equity Selection"*](https://doi.org/10.1080/14697688.2015.1101149) by Liu, Mulvey, Zhao [2015]. I take a universe of assets and create a "Risk Network" of the latent risk-factors in our given universe- advantageous for TFSA investing since the universe is well-defined. A cluster defines assets linked to similar latent factors. I identified four dominant risk-clusters: SA Property, SA Equities, Developed Market Equities, and Modified SA Equities (Islamic banking and inflation protection on SA Equities). 

> The assets that are in clusters of one are considered "risk-independent". They are the most orthogonal to our systematic risk-factors and thus are easy pickings for anyone looking for instant diversification.

<br /> 

###### RESULTS AND INTERPRETATION: 
> My initial hypothesis was that this method would reduce fund performance, but improve risk-adjusted performance. Instead I found a marginal improvement (30 basis points) in performance and significant improvement (10%) in the Sharpe Ratio distribution. Furthermore, the treated portfolios had improved downside-risk performance metrics.
<br /> 

###### SHORTCOMINGS:
> Data, Data, Data. 
<br /> 

> Large "pops" in price data add fat tails to any distribution and heavily skew the results of any parametric modelling methods. This does not affect the graphical model's performance, but rather the means and standard deviation estimates are corrupted. I attempted to stay away from any methods that required expected returns, volatility or covariance wherever possible and thus constructed equally weighted portfolios after the graphical model stage however, the outliers still skewed a lot of the findings


### Steps:
#### 1)  [Data Collection and Processing](#0)
- Scrape TSFA assets available on the [EasyEquities](https://etfs.easyequities.co.za/finder) platform then download price data through the [Yahoo Finance](https://finance.yahoo.com) API

#### 2) [Machine Learning Model](#1)
- Use a sparse covariance matrix to determine significant covariance relations amongst assets and ignore the rest. Much like a social network, assets exposed to similar risk are more likely to be connected to other assets in the 'Risk Network'. Assets in a cluster of one are referred to as Risk-Independent assets

#### 3)  [Test the 'Pure Portfolio' of Risk-Independent assets](#2)
- Build equal risk-contribution and equally weighted portfolios

#### 4) [Experiment](#3)
- Use random portfolio construction to test the impact of adding our Risk-Independent assets
