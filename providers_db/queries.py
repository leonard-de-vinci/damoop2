import sqlite3

DATABASE = "providers.db"

# prices only has ids, so we join the other tables to get the real names (provider, city, server)
ALL_CONFIGURATIONS = """
    SELECT providers.name, regions.city, regions.code, instance_types.name,
           instance_types.vcpu, instance_types.ram_gb, prices.usd_per_hour
    FROM prices
    JOIN regions ON prices.region_id = regions.region_id
    JOIN instance_types ON prices.instance_id = instance_types.instance_id
    JOIN providers ON regions.provider_id = providers.provider_id
"""


def print_rows(rows):
    print(f"{'Provider':<8} {'City':<10} {'Region':<20} {'Server':<10} {'vCPU':>4} {'RAM GB':>6} {'USD/hour':>9}")
    for provider, city, code, server, vcpu, ram, price in rows:
        print(f"{provider:<8} {city:<10} {code:<20} {server:<10} {vcpu:>4} {ram:>6g} {price:>9.4f}")


def show_all_configurations(connection):
    rows = connection.execute(ALL_CONFIGURATIONS + " ORDER BY regions.city, prices.usd_per_hour").fetchall()
    print_rows(rows)


# answers "this data center + this server type", what was asked in the last meeting
def find_configuration(connection, city, server):
    query = ALL_CONFIGURATIONS + " WHERE regions.city = ? AND instance_types.name = ?"
    rows = connection.execute(query, (city, server)).fetchall()
    print_rows(rows)


# cheapest server of a given size in a city, LIMIT 1 keeps only the first row after sorting by price
def cheapest_in_city(connection, city, vcpu, ram_gb):
    query = ALL_CONFIGURATIONS + """
        WHERE regions.city = ? AND instance_types.vcpu = ? AND instance_types.ram_gb = ?
        ORDER BY prices.usd_per_hour
        LIMIT 1
    """
    rows = connection.execute(query, (city, vcpu, ram_gb)).fetchall()
    print_rows(rows)


def show_sources(connection):
    rows = connection.execute("SELECT source_id, title, url, retrieved_on FROM sources").fetchall()
    for source_id, title, url, retrieved_on in rows:
        print(f"[{source_id}] {title}, {url} (retrieved {retrieved_on})")


# this part only runs when we do python3 queries.py
if __name__ == "__main__":
    connection = sqlite3.connect(DATABASE)

    print("\n1. All configurations\n")
    show_all_configurations(connection)

    print("\n2. One data center and one server: Frankfurt, t3.medium\n")
    find_configuration(connection, "Frankfurt", "t3.medium")

    print("\n3. Cheapest 2 vCPU / 4 GB server in Paris\n")
    cheapest_in_city(connection, "Paris", 2, 4)

    print("\n4. Sources\n")
    show_sources(connection)

    connection.close()