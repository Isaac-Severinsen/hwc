# hwc
Hot Water Cylinder Demand Response
This is a work in progress for controlling a hot water cylinder relative to real time prices. 

the file: `get_prices.py` has:

- Authorisation through OAuth
- API connection to WITS to retrieve real-time pricing

the file: `test.py` has:
 - A test to determine what a reasonable fixed maximum price would be
 
![](fixed_price_test.png)

You can see from this that a surprisingly high minimum is required to ensure 3 hours of operation per day for >90% of the year.
It is not clear in this but an non-peak periods throughout the year we would effectively be allowing the cylidner to run 24/7
This is price inefficient as we could be running at $0/MWh if we had a better algorithm.

 - A test to determine what a fixed maximum price would be if it varied throughout the year
 
![](variable_price_test2.png)

You can see here that using a variable maximum price is much more efficient.
In most periods a reasonable price cap of ~100 $/MWh is feasible, with periods above the best fixed rate rare.
The issue with this is that we cannot predetermine this price cap. You can see that in 2021 and 2022 prices increased in the first half of the year but this was not seen in 2020 (COVID?) or 2023 (wet?).

The best method would be to identify daily variations and target the lowest price every calendar day
