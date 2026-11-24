import os
import sqlite3

DATABASE = "providers.db"

# (source_id, title, url, retrieved_on)
sources = [
    (1, "Azure B1s pricing (azurespeed.com, data from Azure Retail Prices API)", "https://www.azurespeed.com/AzureVmPricing/Standard_B1s", "2026-11-23"),
    (2, "Azure B2s pricing (azurespeed.com, data from Azure Retail Prices API)", "https://www.azurespeed.com/AzureVmPricing/Standard_B2s", "2026-11-23"),
    (3, "AWS t3.medium pricing, Paris (doit.com)", "https://www.doit.com/compute/spot/eu-west-3/t3.medium", "2026-11-23"),
    (4, "AWS t3.medium pricing, Frankfurt (doit.com)", "https://www.doit.com/compute/spot/eu-central-1/t3.medium", "2026-11-23"),
    (5, "AWS t3.medium pricing, London (doit.com)", "https://www.doit.com/compute/spot/eu-west-2/t3.medium", "2026-11-23"),
    (6, "GCP e2-medium pricing (gcloud-compute.com)", "https://gcloud-compute.com/e2-medium.html", "2026-11-23"),
    (7, "Azure bandwidth pricing", "https://azure.microsoft.com/en-us/pricing/details/bandwidth/", "2026-11-09"),
    (8, "AWS EC2 on-demand pricing, data transfer section", "https://aws.amazon.com/ec2/pricing/on-demand/", "2026-11-09"),
    (9, "Google Cloud VPC network pricing", "https://cloud.google.com/vpc/network-pricing", "2026-11-09"),
    (10, "Google Cloud region carbon data 2025", "https://github.com/GoogleCloudPlatform/region-carbon-info", "2026-11-23"),
    (11, "Cloud Carbon Footprint methodology", "https://www.cloudcarbonfootprint.org/docs/methodology/", "2026-11-09"),
]

# (provider_id, name, watts_per_vcpu, pue, source_id)
# watts_per_vcpu = min + 50% of (max - min) from CCF, ex azure: 0.78 + 0.5 * (3.76 - 0.78) = 2.27
providers = [
    (1, "Azure", 2.27, 1.185, 11),
    (2, "AWS", 2.12, 1.135, 11),
    (3, "GCP", 2.485, 1.1, 11),
]

# (region_id, provider_id, code, city, grid_gco2_kwh, source_id)
# grid values come from google's dataset, I used the same value for the 3 providers in the same city
# since they use the same electricity grid
regions = [
    (1, 1, "francecentral", "Paris", 15.83, 10),
    (2, 1, "germanywestcentral", "Frankfurt", 275.81, 10),
    (3, 1, "uksouth", "London", 109.39, 10),
    (4, 2, "eu-west-3", "Paris", 15.83, 10),
    (5, 2, "eu-central-1", "Frankfurt", 275.81, 10),
    (6, 2, "eu-west-2", "London", 109.39, 10),
    (7, 3, "europe-west9", "Paris", 15.83, 10),
    (8, 3, "europe-west3", "Frankfurt", 275.81, 10),
    (9, 3, "europe-west2", "London", 109.39, 10),
]

# (instance_id, provider_id, name, vcpu, ram_gb)
# B1s is there because damoop was calibrated on it, the other 3 are all 2 vCPU / 4 GB so we can compare them
instance_types = [
    (1, 1, "B1s", 1, 1),
    (2, 1, "B2s", 2, 4),
    (3, 2, "t3.medium", 2, 4),
    (4, 3, "e2-medium", 2, 4),
]

# (price_id, instance_id, region_id, usd_per_hour, source_id)
# ex: row 8 = t3.medium (instance 3) in AWS Frankfurt (region 5) costs 0.048 $/h
prices = [
    (1, 1, 1, 0.0118, 1),
    (2, 1, 2, 0.0120, 1),
    (3, 1, 3, 0.0118, 1),
    (4, 2, 1, 0.0472, 2),
    (5, 2, 2, 0.0480, 2),
    (6, 2, 3, 0.0472, 2),
    (7, 3, 4, 0.0472, 3),
    (8, 3, 5, 0.0480, 4),
    (9, 3, 6, 0.0472, 5),
    (10, 4, 7, 0.0389, 6),
    (11, 4, 8, 0.0432, 6),
    (12, 4, 9, 0.0432, 6),
]

# (transfer_id, provider_id, transfer_type, usd_per_gb, source_id)
# internet = price after the free tier
transfer_prices = [
    (1, 1, "between_regions", 0.02, 7),
    (2, 1, "internet", 0.087, 7),
    (3, 2, "between_regions", 0.02, 8),
    (4, 2, "internet", 0.09, 8),
    (5, 3, "between_regions", 0.02, 9),
    (6, 3, "internet", 0.12, 9),
]


def create_database():
    # delete the old db so we don't get duplicate rows if we run the script twice
    if os.path.exists(DATABASE):
        os.remove(DATABASE)

    connection = sqlite3.connect(DATABASE)
    # sqlite doesn't check foreign keys by default, this turns it on
    connection.execute("PRAGMA foreign_keys = ON")

    with open("schema.sql") as file:
        connection.executescript(file.read())

    # the ? are filled with the values of each tuple, executemany does it for every row of the list
    # order matters: sources and providers first because the other tables point to them
    connection.executemany("INSERT INTO sources VALUES (?, ?, ?, ?)", sources)
    connection.executemany("INSERT INTO providers VALUES (?, ?, ?, ?, ?)", providers)
    connection.executemany("INSERT INTO regions VALUES (?, ?, ?, ?, ?, ?)", regions)
    connection.executemany("INSERT INTO instance_types VALUES (?, ?, ?, ?, ?)", instance_types)
    connection.executemany("INSERT INTO prices VALUES (?, ?, ?, ?, ?)", prices)
    connection.executemany("INSERT INTO transfer_prices VALUES (?, ?, ?, ?, ?)", transfer_prices)

    connection.commit()
    connection.close()


create_database()
print("Database created:", DATABASE)
print(len(sources), "sources,", len(providers), "providers,", len(regions), "regions,",
      len(instance_types), "instance types,", len(prices), "prices,", len(transfer_prices), "transfer prices")