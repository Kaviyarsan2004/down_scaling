import pandas as pd

# Step 1: Data Preparation
# Load capacity.csv, cluster assignments, and LCOE dataset
capacity_df = pd.read_csv(r"C:\Users\kavi5\Enhancing_Resilient_Solar_Power\git\down_scaling\sample_data\capacity.csv")
cluster_assignments_df = pd.read_csv(r"C:\Users\kavi5\Enhancing_Resilient_Solar_Power\git\down_scaling\sample_data\BASN_UtilityPV_Class1_Moderate__site_cluster_assignments.csv")
lcoe_df = pd.read_csv(r"C:\Users\kavi5\Enhancing_Resilient_Solar_Power\git\down_scaling\sample_data\solar_lcoe_conus_26_zone.csv")

# Filter capacity dataframe for the specific zone and technology
zone_number = 26
technology = "utilitypv"
zone_capacity_df = capacity_df[(capacity_df["Zone"] == zone_number) & (capacity_df["Resource"].str.contains(technology))]

# Extract cluster number from Resource column in capacity dataframe
zone_capacity_df["Cluster"] = zone_capacity_df["Resource"].str.extract(r'(\d+)_anyQual')
print(zone_capacity_df)
# Step 2: Cluster Analysis
clusters = zone_capacity_df["Cluster"].unique()

cluster_df = pd.DataFrame({"Cluster": clusters})


# Step 3: Selection Process
selected_cpas = []

for cluster in clusters:
    # Filter cluster assignments dataframe for the current cluster
    cluster_cpas = cluster_assignments_df[cluster_assignments_df["cluster"] == int(cluster)]["cpa_id"]

    # Filter LCOE dataframe for the CPA_IDs in the current cluster
    cluster_lcoe_df = lcoe_df[lcoe_df["CPA_ID"].isin(cluster_cpas)]
    
    # Sort the dataframe based on LCOE values
    cluster_lcoe_df = cluster_lcoe_df.sort_values(by="lcoe")
    # Calculate the new capacity for the current cluster
    cluster_capacity = zone_capacity_df[zone_capacity_df["Cluster"] == cluster]["NewCap"].iloc[0]
   
    
    # Select CPAs based on cumulative capacity and LCOE values
    cumulative_capacity = 0
    for index, row in cluster_lcoe_df.iterrows():
        cumulative_capacity += row["cpa_mw"]
        selected_cpas.append(row)
        if cumulative_capacity >= cluster_capacity:
            break
print(selected_cpas)
# Step 4: Output
selected_cpas_df = pd.DataFrame(selected_cpas)
selected_cpas_df ["Zone"]=zone_number
# Set index to None for both DataFrames
selected_cpas_df.reset_index(drop=True, inplace=True)
cluster_df.reset_index(drop=True, inplace=True)
selected_cpas_df["Cluster"] = cluster_df["Cluster"]

# Save selected CPAs to a CSV file
selected_cpas_df.to_csv(f"selected_cpas_zone{zone_number}_{technology}.csv", index=False)

# Display selected CPAs
# print(selected_cpas_df)
