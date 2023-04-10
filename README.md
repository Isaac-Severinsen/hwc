# hwc
Hot Water Cylinder Demand Response
This is a work in progress for controlling a hot water cylinder relative to real time prices. 

the file: `get_prices.py` has:

- Authorisation through OAuth
- API connection to WITS to retrieve real-time pricing

the file: `test.py` has:
 - A test to determine what a reasonable fixed maximum price would be
 
![](fixed_price_test.png)

 - A test to determine what a fixed maximum price would be if it varied throughout the year
 
![](variable_price_test2.png)
