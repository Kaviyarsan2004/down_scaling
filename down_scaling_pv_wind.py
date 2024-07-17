import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import geopandas as gpd
from shapely.geometry import Point, polygon
import os
from shapely import wkt
import matplotlib.pyplot as plt

capacity_file = r"C:\Users\kavi5\iter2.csv"#path to cluster specific capacity result for solar for each region file
CPAID_cluster_assignment = r"C:\Users\kavi5\Enhancing_Resilient_Solar_Power\git\down_scaling\sample_data\extra_outputs\extra_outputs\NY_Z_C&E_UtilityPV_Class1_Moderate__site_cluster_assignments.csv" #path to map of cluster to cpa connection file
zone_capacity =r"C:\Users\kavi5\filtered_output.csv"#path to LCOE of each cpa and cpa_mw for each cpa file
result = 'selected_cpas_utilitypv.csv'

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


def randromly_select_cluster( df : pd.DataFrame, tech: str):
    
    pop_density_ranges ={ "utilitypv": [
        (0, 5, 0.2),
        (5, 10, 0.1),
        (10, 30, 0.05),
        (30, 40, 0.025),
        (40, 60, 0.02),
        (60, float('inf'), 0)
    ],
    "onshore_wind":[
         (0, 1, 0.6),
        (1, 3, 0.2),
        (3, 5, 0.1),
        (5, 10, 0.05),
        (10, float('inf'), 0)
    ]
    }
    cluster_factor = {"utilitypv": 1.1, "onshore_wind": 6}
    n_clusters = max(1, int(len(df) // cluster_factor[tech]))

    coords = df[['latitude', 'longitude']].values
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(coords)
    Kmeans =  MiniBatchKMeans(n_clusters=n_clusters, batch_size=10000, random_state=0)
    Kmeans=Kmeans.fit(scaled_data)

    df['cluster'] = Kmeans.labels_

    sampled_dfs = []
    for min_popden, max_popden, sample_frac in pop_density_ranges[tech]:

        df_subset = df[(df['m_popden'] >= min_popden) & (df['m_popden'] < max_popden)]
        selected_clust = (
            pd.DataFrame(df_subset["cluster"].unique())
            .sample(frac=sample_frac, replace=False)
            .rename(columns={0: "cluster"})
        )
        final_df = df_subset.loc[
            df_subset["cluster"].isin(selected_clust["cluster"])
        ]
        sampled_dfs.append(final_df)

    # Combine all the sampled and merged dataframes
    merged_df = pd.concat(sampled_dfs)
    merged_df.reset_index(drop=True, inplace=True)
    return merged_df


def select_lcoe_cpas(
    lcoe_df: pd.DataFrame,
    cluster_assignments_df: pd.DataFrame,
    cluster: int,
    cluster_capacity: float
) -> pd.DataFrame:
    """Select CPAs based on LCOE values and cluster capacity."""
    # Normalize column names to a consistent case (lowercase in this example)
    lcoe_df.columns = lcoe_df.columns.str.lower()
    cluster_cpas = cluster_assignments_df[cluster_assignments_df["cluster"] == cluster]["cpa_id"]
    cluster_lcoe_df = lcoe_df[lcoe_df["cpa_id"].isin(cluster_cpas)]
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
    technology: str,
    output_file: str
) -> None:


    capacity_df = load_capacity_data(capacity_file)
    cluster_assignments_df = load_cluster_assignments(cluster_file)
    lcoe_df = load_lcoe_data(lcoe_file)
    df= randromly_select_cluster(lcoe_df, technology)
    zone_numbers = capacity_df["Zone"].unique()
    selected_cpas = []
    
    for zone_number in zone_numbers:
        zone_capacity_df = capacity_df[(capacity_df["Zone"] == float(zone_number)) & (capacity_df["Resource"].str.contains(technology))].copy()
        if(technology=="utilitypv"):
            zone_capacity_df["Cluster"] = zone_capacity_df["Resource"].str.extract(r'moderate_(\d+)')
        else:
            zone_capacity_df["Cluster"] = zone_capacity_df["Resource"].str.extract(r'onshore_wind_turbine_(\d+)')
        
        
        clusters = zone_capacity_df["Cluster"].unique()
        for cluster in clusters:
            cluster_capacity = zone_capacity_df[zone_capacity_df["Cluster"] == cluster]["NewCap"].iloc[0]
            selected_cpas_df = select_lcoe_cpas(df, cluster_assignments_df, int(cluster), cluster_capacity)
            selected_cpas_df["Zone"] = zone_number
            selected_cpas_df["cluster"]=cluster
            selected_cpas_df["NewCap"] = cluster_capacity
            selected_cpas.append(selected_cpas_df)
    
    
    final_selected_cpas_df = pd.concat(selected_cpas)
    final_selected_cpas_df.reset_index(drop=True, inplace=True)
    final_selected_cpas_df = final_selected_cpas_df[['Zone', 'cluster', 'NewCap'] + [col for col in final_selected_cpas_df.columns if col not in ['Zone', 'Cluster', 'NewCap']]]
    
    # Save selected CPAs to CSV file
    final_selected_cpas_df.to_csv(output_file, index=False)

    cpa_id_df = final_selected_cpas_df
    geometry_df = pd.read_csv(r"C:\Users\kavi5\Enhancing_Resilient_Solar_Power\git\Community-Energy-Compass\data\NY_solar_layer-new.csv")

    # Convert all column names to lowercased
    cpa_id_df.columns = cpa_id_df.columns.str.lower()
    geometry_df.columns = geometry_df.columns.str.lower()

    # Convert the 'geometry' column to a GeoSeries
    geometry_df['geometry'] = geometry_df['geometry'].apply(wkt.loads)
    geometry_gdf = gpd.GeoDataFrame(geometry_df, geometry='geometry')

    # Merge the dataframes on 'cpa_id'
    merged_df = pd.merge(cpa_id_df, geometry_gdf, on='cpa_id')


    df =  merged_df
    column_names=list(df.columns)
    seen = {}
    for i, col in enumerate(column_names):
        if col in seen:
            seen[col] += 1
            column_names[i] = f"{col}_{seen[col]}"
        else:
            seen[col] = 0
    df.columns=column_names
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    # Step 3: Define the coordinate reference system (CRS), if known. Example: WGS84
    gdf.set_crs(epsg=4326, inplace=True)  # WGS84

    # Step 4: Create a new folder to store the shapefile
    output_folder = 'shapefile'
    os.makedirs(output_folder, exist_ok=True)

    # Step 5: Save the GeoDataFrame as a shapefile
    shapefile_path = os.path.join(output_folder, f'downScaled_{technology}.shp')
    gdf.to_file(shapefile_path, driver='ESRI Shapefile')

    print(f"Shapefile saved to {shapefile_path}")

    print(f"Selecetd CPAs as been saved {final_selected_cpas_df.shape}")
    gdf.plot()
    plt.show()

print("Please choose a Technology:")
print("1. Solar")
print("2. wind")

choice = input("Enter the number of your choice (1 or 2): ")
if choice == '1':
    technology="utilitypv"
elif choice == '2':
    technology="onshore_wind"
optimize_program(capacity_file, CPAID_cluster_assignment, zone_capacity, technology , result)


