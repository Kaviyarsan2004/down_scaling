import pandas as pd
from typing import List, Tuple
from sklearn.cluster import KMeans
import numpy as np

def load_capacity_data(capacity_file: str) -> pd.DataFrame:
    """Load capacity data from the given file."""
    capacity_df = pd.read_csv(capacity_file)
    return capacity_df

def load_cluster_assignments(cluster_file: str) -> pd.DataFrame:
    """Load cluster assignments from the given file."""
    cluster_assignments_df = pd.read_csv(cluster_file)
    return cluster_assignments_df

def load_lcoe_data(lcoe_file: str) -> pd.DataFrame:
    """Load LCOE data from the given file."""
    lcoe_df = pd.read_csv(lcoe_file)
    return lcoe_df

def select_cpas(
    lcoe_df: pd.DataFrame,
    cluster_assignments_df: pd.DataFrame,
    cluster: int,
    cluster_capacity: float
) -> pd.DataFrame:
    """Select CPAs based on LCOE values and cluster capacity."""
    cluster_cpas = cluster_assignments_df[cluster_assignments_df["cluster"] == cluster]["cpa_id"]
    cluster_lcoe_df = lcoe_df[lcoe_df["CPA_ID"].isin(cluster_cpas)]
    cluster_lcoe_df = cluster_lcoe_df.sort_values(by="lcoe")
    
    cumulative_capacity = 0
    selected_cpas = []
    for _, row in cluster_lcoe_df.iterrows():
        cumulative_capacity += row["cpa_mw"]
        selected_cpas.append(row)
        if cumulative_capacity >= cluster_capacity:
            break
    
    selected_cpas_df = pd.DataFrame(selected_cpas)
    return selected_cpas_df

def optimize_program(
    capacity_file: str,
    cluster_file: str,
    lcoe_file: str,
    zone_number: int,
    technology: str,
    output_file: str
) -> None:
    """Optimized version of Program 1."""
    capacity_df = load_capacity_data(capacity_file)
    cluster_assignments_df = load_cluster_assignments(cluster_file)
    lcoe_df = load_lcoe_data(lcoe_file)
    
    zone_capacity_df = capacity_df[(capacity_df["Zone"] == zone_number) & (capacity_df["Resource"].str.contains(technology))]
    zone_capacity_df["Cluster"] = zone_capacity_df["Resource"].str.extract(r'(\d+)_anyQual')
    
    clusters = zone_capacity_df["Cluster"].unique()
    selected_cpas = []
    for cluster in clusters:
        cluster_capacity = zone_capacity_df[zone_capacity_df["Cluster"] == cluster]["NewCap"].iloc[0]
        selected_cpas_df = select_cpas(lcoe_df, cluster_assignments_df, int(cluster), cluster_capacity)
        selected_cpas_df["Zone"] = zone_number
        selected_cpas.append(selected_cpas_df)
    
    final_selected_cpas_df = pd.concat(selected_cpas)
    final_selected_cpas_df.reset_index(drop=True, inplace=True)
    
    # Save selected CPAs to CSV file
    final_selected_cpas_df.to_csv(output_file, index=False)

# Example usage:
optimize_program("capacity.csv", "BASN_UtilityPV_Class1_Moderate__site_cluster_assignments.csv", "solar_lcoe_conus_26_zone.csv", 26, "utilitypv", "selected_cpas.csv")
