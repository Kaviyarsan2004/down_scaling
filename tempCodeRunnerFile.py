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