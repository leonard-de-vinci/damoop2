-- every table that has a number in it points to sources, so we always know where a value comes from
CREATE TABLE sources (
    source_id     INTEGER PRIMARY KEY,
    title         TEXT NOT NULL,
    url           TEXT NOT NULL,
    retrieved_on  TEXT NOT NULL      -- date I collected the value, prices change so this matters
);

CREATE TABLE providers (
    provider_id     INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    watts_per_vcpu  REAL NOT NULL,   -- avg power of 1 vCPU at 50% load (Cloud Carbon Footprint values)
    pue             REAL NOT NULL,   -- extra energy used by the data center (cooling etc), 1.185 = +18.5%
    source_id       INTEGER REFERENCES sources(source_id)
);

CREATE TABLE regions (
    region_id      INTEGER PRIMARY KEY,
    provider_id    INTEGER NOT NULL REFERENCES providers(provider_id),
    code           TEXT NOT NULL,    -- name the provider uses, ex: francecentral, eu-west-3
    city           TEXT NOT NULL,
    grid_gco2_kwh  REAL NOT NULL,    -- grams of CO2 per kWh of electricity in that country/region
    source_id      INTEGER REFERENCES sources(source_id)
);

-- server sizes, no price here because the price depends on the region
CREATE TABLE instance_types (
    instance_id  INTEGER PRIMARY KEY,
    provider_id  INTEGER NOT NULL REFERENCES providers(provider_id),
    name         TEXT NOT NULL,
    vcpu         INTEGER NOT NULL,
    ram_gb       REAL NOT NULL
);

-- main table: price of one server type in one region (on-demand, linux)
CREATE TABLE prices (
    price_id      INTEGER PRIMARY KEY,
    instance_id   INTEGER NOT NULL REFERENCES instance_types(instance_id),
    region_id     INTEGER NOT NULL REFERENCES regions(region_id),
    usd_per_hour  REAL NOT NULL,
    source_id     INTEGER REFERENCES sources(source_id)
);

-- cost of sending data, same for the whole provider in europe so linked to providers not regions
CREATE TABLE transfer_prices (
    transfer_id    INTEGER PRIMARY KEY,
    provider_id    INTEGER NOT NULL REFERENCES providers(provider_id),
    transfer_type  TEXT NOT NULL,    -- between_regions or internet
    usd_per_gb     REAL NOT NULL,
    source_id      INTEGER REFERENCES sources(source_id)
);