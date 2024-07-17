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

capacity_file = r"C:\Users\kavi5\Enhancing_Resilient_Solar_Power\git\down_scaling\sample_data\combined_capacity_MGA_max.csv"#path to cluster specific capacity result for solar for each region file
CPAID_lcoe = r"C:\Users\kavi5\Enhancing_Resilient_Solar_Power\git\down_scaling\sample_data\solar_lcoe_ipm_metro_county.csv" #path to map of cluster to cpa connection file
result = 'selected_cpas_utilitypv.csv'

folder_path = r"C:\Users\kavi5\Enhancing_Resilient_Solar_Power\git\down_scaling\sample_data\extra_outputs\extra_outputs" 

def load_cluster_assignments(capacity_file: str) -> pd.DataFrame:
    """Load capacity data from the given file."""
    capacity_df = pd.read_csv(capacity_file)
    return capacity_df

def load_capacity_data(cluster_file: str) -> pd.DataFrame:
    """Load cluster assignments from the given file."""
    df = pd.read_csv(cluster_file)
    generator_data=pd.read_csv(r"C:\Users\kavi5\Enhancing_Resilient_Solar_Power\git\down_scaling\sample_data\Generators_data.csv")

    iteration_number = 0
    in_iteration = False

    # Iterate through the DataFrame and add the iteration number
    iteration_numbers = []
    for index, row in df.iterrows():
        if 'NENG_Rest_battery_moderate_0' in row.values:
            iteration_number += 1
            in_iteration = True
        elif 'Total' in row.values:
            in_iteration = False

        if in_iteration or ('Total' in row.values and not in_iteration):
            iteration_numbers.append(iteration_number)
        else:
            iteration_numbers.append(None)

    df['iter'] = iteration_numbers
    df = df[df['Resource'].str.contains('_landbasedwind|utilitypv', case=False, na=False)]
    
    condition_utilitypv = df['Resource'].str.contains('utilitypv', case=False, na=False)

    df.loc[condition_utilitypv, "Cluster"] = df.loc[condition_utilitypv, "Resource"].str.extract(r'moderate_(\d+)', expand=False)
    df.loc[~condition_utilitypv, "Cluster"] = df.loc[~condition_utilitypv, "Resource"].str.extract(r'landbasedwind_class1_moderate_(\d+)', expand=False)

    generator_data = generator_data.drop_duplicates(subset=['Zone'], keep='first')
    df = pd.merge(df, generator_data[['Zone', 'region']], on=['Zone'], how='left')
    df['technology'] = df['Resource'].apply(lambda x: 'UtilityPV_Class1_Moderate_' if 'utilitypv' in x.lower() else 'LandbasedWind')
    df.to_csv("capacity_alter.csv")
    return df

def load_lcoe_data(lcoe_file: str) -> pd.DataFrame:
    """Load LCOE data from the given file."""
    lcoe_df = pd.read_csv(lcoe_file)
    return lcoe_df


def randomly_select_cluster( df : pd.DataFrame, tech: str):
    
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

    df['cluster_mean'] = Kmeans.labels_

    sampled_dfs = []
    for min_popden, max_popden, sample_frac in pop_density_ranges[tech]:

        df_subset = df[(df['m_popden'] >= min_popden) & (df['m_popden'] < max_popden)]
        selected_clust = (
            pd.DataFrame(df_subset["cluster_mean"].unique())
            .sample(frac=sample_frac, replace=False)
            .rename(columns={0: "cluster_mean"})
        )
        final_df = df_subset.loc[
            df_subset["cluster_mean"].isin(selected_clust["cluster_mean"])
        ]
        sampled_dfs.append(final_df)

    # Combine all the sampled and merged dataframes
    merged_df = pd.concat(sampled_dfs)
    merged_df.reset_index(drop=True, inplace=True)
    return merged_df

def find_and_read_matching_file(region, technology):
    
    # Construct the pattern to search for in file names
    pattern = f"{region}_{technology.split('_')[0]}"
    
    # Check all files in the folder
    matching_files = [file for file in os.listdir(folder_path) if pattern in file]
    
    if matching_files:
        file_name = matching_files[0] 
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        return df
    else:
        return pd.DataFrame(), None  # Return an empty DataFrame and None if no matching file found


def select_lcoe_cpas(
    lcoe_df: pd.DataFrame,
    cluster: int,
    cluster_capacity: float,
    technology: str,
    region:str
) -> pd.DataFrame:
    """Select CPAs based on LCOE values and cluster capacity."""
    # Normalize column names to a consistent case (lowercase in this example)
    lcoe_df.columns = lcoe_df.columns.str.lower()
    cluster_assignments_df=find_and_read_matching_file(region,technology)
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
    lcoe_file: str,
    output_file: str
) -> None:


    capacity_df = load_capacity_data(capacity_file)
    # lcoe_df = load_lcoe_data(lcoe_file)
    df= load_lcoe_data(lcoe_file)
    iter = capacity_df["iter"].unique()
    selected_cpas = []
    
    for iter_number in iter:
        zone_capacity_df = capacity_df[(capacity_df["iter"] == float(iter_number))].copy()
        iter_selected_cpas = []
        for intex, row in zone_capacity_df.iterrows():
            cluster_capacity = row["NewCap"]
            cluster=row['Cluster']
            technology=row['technology']
            region=row['region']
            selected_cpas_df = select_lcoe_cpas(df, int(cluster), cluster_capacity,technology,region)
            selected_cpas_df["Zone"] = row['Zone']
            selected_cpas_df["technology"] = row['technology'] 
            selected_cpas_df["cluster"]=row['Cluster']
            selected_cpas_df["NewCap"] = row["NewCap"]
            selected_cpas_df["iter"] = row['iter']
            iter_selected_cpas.append(selected_cpas_df)

            
        
       # Convert the list of dataframes to a single dataframe for this iteration
        iter_selected_cpas_df = pd.concat(iter_selected_cpas, ignore_index=True)

        # Filtering and random selection for UtilityPV_Class1_Moderate_
        selected_solar_cpas_df = iter_selected_cpas_df[iter_selected_cpas_df['technology'] == 'UtilityPV_Class1_Moderate_']
        if len(selected_solar_cpas_df.index) > 1:
            selected_solar_cpas_df =  randomly_select_cluster(selected_solar_cpas_df, 'utilitypv')
        selected_cpas.append(selected_solar_cpas_df)

        # Filtering and random selection for LandbasedWind
        selected_wind_cpas_df = iter_selected_cpas_df[iter_selected_cpas_df['technology'] == 'LandbasedWind']
        if len(selected_wind_cpas_df.index) > 1:
            selected_wind_cpas_df = randomly_select_cluster(selected_wind_cpas_df, 'onshore_wind')
        selected_cpas.append(selected_wind_cpas_df)

    # Convert the list of dataframes to a single dataframe after the loop if needed
    final_selected_cpas_df = pd.concat(selected_cpas, ignore_index=True)
        
    # final_selected_cpas_df = final_selected_cpas_df[['Zone', 'cluster', 'NewCap'] + [col for col in final_selected_cpas_df.columns if col not in ['Zone', 'Cluster', 'NewCap']]]
    
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


optimize_program(capacity_file, CPAID_lcoe, result)


