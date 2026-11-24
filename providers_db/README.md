# Cloud providers database (WP2)

Small SQLite database with cloud server configurations (Azure, AWS, GCP) in Europe. Every value has a source link. The idea is to use it to get the provider dependent constants of the DaMoOp cost model instead of the fixed ones in the jar.

## How to run

Only needs Python 3.

```
python3 create_database.py
python3 queries.py
python3 damoop_constants.py
```

create_database.py has to be run first since it builds providers.db. Running it again deletes the old db and rebuilds it.

## Files

- schema.sql: creates the 6 tables
- create_database.py: creates providers.db and inserts the data
- queries.py: some example queries (all configurations, one city + one server, cheapest server in a city, sources)
- damoop_constants.py: calculates the DaMoOp constants for a provider, city and server
- database_schema.drawio: the schema, open it with app.diagrams.net

## Tables

- providers: Azure, AWS, GCP + watts per vCPU and PUE (needed for carbon)
- regions: data centers, with the city and the carbon intensity of the electricity there
- instance_types: server sizes (vCPU, RAM)
- prices: price per hour of one server type in one region
- transfer_prices: price per GB sent between regions or to the internet
- sources: title, link and date for every source

## DaMoOp constants

- cssdF (EUR per server per day) = price per hour x 24 x 0.86
- csrvE (kg CO2 per server per day) = energy used in a day x grid carbon
- ccomF (EUR per GB) = transfer price between regions x 0.86
- ccomE (kg CO2 per GB) = 0.001 kWh per GB x grid carbon

## What's in it for now

Paris, Frankfurt and London (all 3 providers have a data center there), with 4 servers: Azure B1s (what DaMoOp was calibrated on) and 3 servers of the same size (2 vCPU / 4 GB) to compare providers. Prices are on-demand Linux in USD. Grid carbon is from Google's 2025 data, same value for all providers in the same city.

## Limits

- prices were collected by hand for now, the scraper will come later
- carbon only counts electricity, not the manufacturing of the hardware
