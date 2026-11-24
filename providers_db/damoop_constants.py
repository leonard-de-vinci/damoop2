import sqlite3

DATABASE = "providers.db"

USD_TO_EUR = 0.86          # rate on 17 nov 2026, damoop works in euros
WATTS_PER_GB_RAM = 0.392   # from cloud carbon footprint (0.000392 kWh per GB per hour)
NETWORK_KWH_PER_GB = 0.001 # energy to send 1 GB over the network, also from CCF

# values that are hard coded right now in damoop's CostCalculation.jar (calibrated on azure B1s france)
# names: F = financial (euros), E = environmental (kg CO2), srv = server, com = network
# cssdF is actually the price per server per day even if the name says ssd, that's how it is in their code
DAMOOP_CURRENT = {
    "cssdF": 4.3512,
    "csrvE": 0.87671,
    "ccomF": 0.019,
    "ccomE": 0.011,
}


def get_constants(connection, provider, city, server):
    query = """
        SELECT prices.usd_per_hour, instance_types.vcpu, instance_types.ram_gb,
               regions.grid_gco2_kwh, providers.watts_per_vcpu, providers.pue,
               transfer_prices.usd_per_gb
        FROM prices
        JOIN regions ON prices.region_id = regions.region_id
        JOIN instance_types ON prices.instance_id = instance_types.instance_id
        JOIN providers ON regions.provider_id = providers.provider_id
        JOIN transfer_prices ON transfer_prices.provider_id = providers.provider_id
        WHERE providers.name = ? AND regions.city = ? AND instance_types.name = ?
          AND transfer_prices.transfer_type = 'between_regions'
    """
    row = connection.execute(query, (provider, city, server)).fetchone()
    if row is None:
        print("No data for", provider, city, server)
        return None

    usd_per_hour, vcpu, ram_gb, grid, watts_per_vcpu, pue, usd_per_gb = row

    # power of the server, then energy for a full day (/1000 to go from Wh to kWh), times pue
    watts = vcpu * watts_per_vcpu + ram_gb * WATTS_PER_GB_RAM
    kwh_per_day = watts * 24 / 1000 * pue

    # grid is in grams per kWh so /1000 to get kg like damoop
    return {
        "cssdF": usd_per_hour * 24 * USD_TO_EUR,         # euros per server per day
        "csrvE": kwh_per_day * grid / 1000,              # kg CO2 per server per day
        "ccomF": usd_per_gb * USD_TO_EUR,                # euros per GB sent
        "ccomE": NETWORK_KWH_PER_GB * grid / 1000,       # kg CO2 per GB sent
    }


def print_constants(name, constants):
    print(f"{name:<26} {constants['cssdF']:>10.4f} {constants['csrvE']:>10.5f} "
          f"{constants['ccomF']:>10.4f} {constants['ccomE']:>10.6f}")


if __name__ == "__main__":
    connection = sqlite3.connect(DATABASE)

    print(f"\n{'':<26} {'cssdF':>10} {'csrvE':>10} {'ccomF':>10} {'ccomE':>10}")
    print(f"{'':<26} {'EUR/day':>10} {'kg/day':>10} {'EUR/GB':>10} {'kg/GB':>10}")
    print_constants("DaMoOp today", DAMOOP_CURRENT)

    choices = [
        ("Azure", "Paris", "B1s"),
        ("Azure", "Paris", "B2s"),
        ("AWS", "Paris", "t3.medium"),
        ("GCP", "Paris", "e2-medium"),
        ("GCP", "London", "e2-medium"),
        ("GCP", "Frankfurt", "e2-medium"),
    ]
    for provider, city, server in choices:
        constants = get_constants(connection, provider, city, server)
        if constants is not None:
            print_constants(f"{provider} {server} {city}", constants)

    connection.close()