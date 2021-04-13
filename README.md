# Pricing-American-Options-through-Reinforcement-Learning

This project aims to price auto-callable options using reinforcement learning methods.
First I applied Black-Scholes and Heston model to simulate AAPL inc. stock price.
I have modified Least Squares Policy Iteration (LSPI) and Fitted Q Iteration (FQI) using python to fit data type and option derivatives used in the project.
After extracted the last 2 years of Apple Inc. historical price data using Excel and Python Pandas, I conducted back-testing of the previous methods. I calculated the Profit and Loss (P&L) of each approach. 
This analysis shows a significant 330% hedging performance improvement using LSPI and FQI compared to the classic Least squares
Monte Carlo method (LSM).
