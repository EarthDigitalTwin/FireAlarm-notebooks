from datetime import datetime, timedelta
from typing import List, Tuple, Iterable, Union
import pandas as pd
import numpy as np
import requests
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.dates as mdates
import io
import folium
from branca.colormap import LinearColormap

from IPython.display import Image
from PIL import Image as I
from PIL import ImageDraw, ImageFont

from glob import glob
import hashlib
import os

QUERY_URL = "https://ideas-digitaltwin.jpl.nasa.gov/insitu/1.0/query_data_doms_custom_pagination?startIndex=0&itemsPerPage=10000"
PROJECT = "air_quality"
CACHE_DIR = ".cache"


def enforce_cache_limit(max_files: int = 40):
    """
    Ensures we never have more than `max_files` cached results.
    Removes the oldest files first, based on modification time.
    """
    # Get all cached pickle files
    files = glob(os.path.join(CACHE_DIR, "*.pkl"))

    # If under the limit, do nothing
    if len(files) <= max_files:
        return

    # Sort files by last modified time (oldest first)
    files.sort(key=lambda x: os.path.getmtime(x))

    # Delete the oldest files beyond the allowed limit
    for old_file in files[: len(files) - max_files]:
        try:
            os.remove(old_file)
            print(f"Deleted old cache file: {old_file}")
        except Exception as e:
            print(f"Failed to delete {old_file}: {e}")


def cache_filename(url: str) -> str:
    """
    Converts a query URL into a unique filename using a SHA-256 hash.
    This ensures the filename is consistent and unique for each query.
    """
    # Make sure the cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Create a unique hash from the URL string
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()

    # Return the full path for the cached file
    return os.path.join(CACHE_DIR, f"{url_hash}.pkl")


def results_to_df(results: list) -> pd.DataFrame:
    """
    Converts a list of JSON-like results from the backend into a cleaned pandas DataFrame.
    Also sets the DataFrame index to datetime for easier time-series handling.
    """
    # Convert the list of result dictionaries to a DataFrame
    df = pd.DataFrame(results)

    # Extract a readable platform name for each row
    df["platform"] = df["platform"].apply(
        lambda x: x["name"] if "name" in x else x["short_name"]
    )

    # Drop columns where all values are NaN
    df = df.dropna(axis=1, how="all")

    # Reorder columns: put 'platform' at the end, drop internal metadata fields
    cols = [c for c in df.columns if c not in ["job_id", "platform_id", "project"]]
    cols = cols[1:] + ["platform"]  # move 'platform' to the end
    df = df[cols]

    # Convert time strings to datetime objects and set as index
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    return df


def data_request(
    platform: Union[str, List[str]],
    start_time: datetime,
    end_time: datetime,
    bounding_box: str,
    provider: str = "SARP",
    force: bool = False,
) -> pd.DataFrame:
    """
    Request in situ data from a remote API. Results are cached locally to avoid repeated downloads.

    Parameters:
        platform (str or list of str): Platform name(s) to include in the query.
        start_time (datetime): Start of the time range to query.
        end_time (datetime): End of the time range to query.
        bounding_box (str): Geographic bounding box (e.g., "lon1,lat1,lon2,lat2").
        provider (str): Data provider. Defaults to "SARP".
        force (bool): If True, bypass cache and force a new download.

    Returns:
        pd.DataFrame: A dataframe containing the requested observation data.
    """

    # Format platform list into comma-separated string for the URL
    if isinstance(platform, str):
        platform_param = platform
    elif isinstance(platform, Iterable):
        platform_param = ",".join(platform)
        
    # Construct the API query URL with all parameters
    initial_url = (
        f'{QUERY_URL}&provider={provider}&project={PROJECT}'
        f'&platform={platform_param}'
        f'&startTime={start_time.strftime("%Y-%m-%dT%H:%M:%SZ")}'
        f'&endTime={end_time.strftime("%Y-%m-%dT%H:%M:%SZ")}'
        f'&bbox={bounding_box}'
    )

    # Generate a filename to use as the cache key for this request
    filename = cache_filename(initial_url)

    # If cached result exists and force is False, load from cache
    if os.path.exists(filename) and not force:
        return pd.read_pickle(filename)
    else:
        # Otherwise, fetch data from API
        results = []
        next_url = initial_url
        
        # Loop through paginated results (if API provides multiple pages)
        while next_url and next_url != "NA":
            print(next_url) # Useful for debugging or observing request flow
            try:
                res = requests.get(next_url)
                res.raise_for_status() # Raises an error for HTTP error codes
                json_results = res.json() # Parse JSON response
            except Exception as e:
                raise ValueError(f"Error fetching/parsing JSON from {next_url}: {e}")

            # Append current page's results
            results.extend(json_results.get("results"))

            # Prepare for next iteration if pagination is present
            next_url = json_results.get("next")
            if next_url == initial_url or not next_url:
                break

        # If no data was returned, raise an error
        if not results:
            raise ValueError(f"No data found for this query: {initial_url}")

        # Convert list of results (dicts) to a DataFrame
        df = results_to_df(results)

        # Enforce local cache size limits and write to cache
        enforce_cache_limit()
        df.to_pickle(filename)

        return df


"""
Functions containing code for the "SARP_demo.ipynb" notebook.
"""


def obtain_picarro_ghg() -> pd.DataFrame:
    """
    Query the FireAlarm system to obtain Greenhouse Gas (GHG) data from the PICARRO sensor.

    The function fetches data in JSON format, converts it to a Pandas DataFrame, replaces fill values with NaN,
    and removes any rows containing missing data.

    Returns:
        pd.DataFrame: Cleaned DataFrame containing PICARRO GHG data for specified platform, time range, and area.
    """
    # Specify the PICARRO platform name
    platform: str = "Dynamic-Aviation-King-Air-(N87Q)--PICARRO-GHG-SLOW-DA"

    # Define the time range for the data query (start and end as datetime objects)
    start_time: datetime = datetime(2024, 7, 2)
    end_time: datetime = datetime(2024, 7, 4)

    # Define the bounding box for spatial filtering: "min_lon,min_lat,max_lon,max_lat"
    bounding_box: str = "-118.75,33.5,-115,34.25"

    # Request data using data_request with platform, time range, and bounding box
    df: pd.DataFrame = data_request(platform, start_time, end_time, bounding_box)

    # Replace fill values (-9999.0) in specific columns with NaN
    nan_cals = ["co2", "ch4", "water_vapor"]
    df[nan_cals] = df[nan_cals].replace(-9999.0, np.nan)

    # Replace CO fill value (-9.999) with NaN since CO was converted to ppm
    df["co"] = df["co"].replace(-9.999, np.nan)

    # Drop all rows with any NaN values to ensure clean data
    df = df.dropna(axis=0, how="any")

    # Return the cleaned DataFrame
    return df


def add_colormap_to_map(m: folium.Map, colormap: LinearColormap, position: str, idx: int):
    """
    Add a branca colormap to a folium map with CSS to position and stack multiple colorbars.

    Args:
        m: Folium map to add the colormap to.
        colormap: Branca LinearColormap instance.
        position: 'bottomleft' or 'bottomright' (only bottomleft used here).
        idx: Index to vertically offset multiple colorbars.
    """
    # Generate HTML for the colorbar
    colorbar_html = colormap._repr_html_()

    # CSS style for fixed position and vertical stacking by index (idx)
    style = f"""
    <style>
        .custom-colorbar-{idx} {{
            position: fixed;
            bottom: {-50 + idx * 70}px;
            left: 10px;
            width: 475px !important;
            z-index: 9999;
            background: rgba(256, 256, 256, 0.5);
            padding: 5px 10px 5px 5px;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
            border-radius: 5px;
            font-size: 12px;
        }}
    </style>
    """

    # Wrap the colorbar html in a div with the above CSS class
    colorbar_div = f"""
    {style}
    <div class="custom-colorbar-{idx}">
        {colorbar_html}
    </div>
    """

    # Create a folium Element and add it to the map's html
    el = folium.Element(colorbar_div)
    m.get_root().html.add_child(el)


def map_picarro_ghg(df: pd.DataFrame) -> folium.Map:
    """
    Create a Folium map with multiple layers showing greenhouse gas (GHG) measurements.

    Each layer represents a different attribute from the PICARRO dataset:
    - Time markers colored by timestamp
    - Altitude markers colored by altitude
    - CO2 concentration markers colored by CO2 values
    - CO concentration markers colored by CO values
    - CH4 concentration markers colored by CH4 values

    Args:
        df (pd.DataFrame): DataFrame containing PICARRO GHG data with columns
            'latitude', 'longitude', 'gps_altitude', 'co2', 'co', 'ch4', and a datetime index.

    Returns:
        folium.Map: Interactive map with GHG data layers and controls.
    """
    m = folium.Map(location=[34, -117.9], zoom_start=10, tiles=None)

    # Add ESRI imagery as basemap
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri — Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, etc.",
        name="Esri Satellite",
        overlay=False,
        control=False,
    ).add_to(m)

    # Establish each layer's color normalization
    timestamps = df.index
    timestamp_seconds = timestamps.astype("int64") // 10**9
    time_norm = colors.Normalize(vmin=timestamp_seconds.min(), vmax=timestamp_seconds.max())
    altitude_norm = colors.Normalize(vmin=df["gps_altitude"].min(), vmax=1500)
    co2_norm = colors.Normalize(vmin=np.nanmin(df["co2"]), vmax=np.nanmax(df["co2"]))
    co_norm = colors.Normalize(vmin=np.nanmin(df["co"]), vmax=np.nanmax(df["co"]))
    ch4_norm = colors.Normalize(vmin=np.nanmin(df["ch4"]), vmax=np.nanmax(df["ch4"]))

    # Establish each layer's feature group
    time_layer = folium.FeatureGroup(name="Time Markers", show=True, overlay=False)
    altitude_layer = folium.FeatureGroup(name="Altitude Markers", show=False, overlay=False)
    co2_layer = folium.FeatureGroup(name="CO₂ Markers", show=False, overlay=False)
    co_layer = folium.FeatureGroup(name="CO Markers", show=False, overlay=False)
    ch4_layer = folium.FeatureGroup(name="CH4 Markers", show=False, overlay=False)

    # Iterate through the rows of the DataFrame
    # Add a marker to each layer
    for idx, row in df.iterrows():
        # Time marker
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color=colors.rgb2hex(cm.jet(time_norm(idx.value // 10**9))),
            fill=True,
            fill_opacity=0.8,
            popup=idx.strftime("%b %d %H:%M:%S"),
        ).add_to(time_layer)

        # Altitude marker 
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color=colors.rgb2hex(cm.YlGnBu(altitude_norm(row["gps_altitude"]))),
            fill=True,
            fill_opacity=0.8,
            popup=f"{row['gps_altitude']:.2f} m",
        ).add_to(altitude_layer)

        # CO2 marker 
        co2_color = colors.rgb2hex(cm.Spectral(co2_norm(row["co2"])))
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color=co2_color,
            fill=True,
            fill_color=co2_color,
            fill_opacity=0.8,
            popup=f"{row['co2']:.1f} ppm",
        ).add_to(co2_layer)

        # CO marker
        co_color = colors.rgb2hex(cm.magma(co_norm(row["co"])))
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color=co_color,
            fill=True,
            fill_color=co_color,
            fill_opacity=0.8,
            popup=f"{row['co']:.1f} ppm",
        ).add_to(co_layer)

        # CH4 marker
        ch4_color = colors.rgb2hex(cm.viridis(ch4_norm(row["ch4"])))
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color=ch4_color,
            fill=True,
            fill_color=ch4_color,
            fill_opacity=0.8,
            popup=f"{row['ch4']:.1f} ppm",
        ).add_to(ch4_layer)

    # Add the layers to the map
    time_layer.add_to(m)
    altitude_layer.add_to(m)
    co2_layer.add_to(m)
    co_layer.add_to(m)
    ch4_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # --- Add colorbars ---

    altitude_colormap = LinearColormap(
        colors=[cm.YlGnBu(i) for i in np.linspace(0, 1, 256)],
        vmin=round(df["gps_altitude"].min(), 1),
        vmax=1500.0,
        caption="Altitude (m)",
    )

    co2_colormap = LinearColormap(
        colors=[cm.Spectral(i) for i in np.linspace(0, 1, 256)],
        vmin=round(np.nanmin(df["co2"]), 1),
        vmax=round(np.nanmax(df["co2"]), 1),
        caption="CO₂ (ppm)",
    )

    co_colormap = LinearColormap(
        colors=[cm.magma(i) for i in np.linspace(0, 1, 256)],
        vmin=round(np.nanmin(df["co"]), 1),
        vmax=round(np.nanmax(df["co"]), 1),
        caption="CO (ppm)",
    )

    ch4_colormap = LinearColormap(
        colors=[cm.viridis(i) for i in np.linspace(0, 1, 256)],
        vmin=round(np.nanmin(df["ch4"]), 1),
        vmax=round(np.nanmax(df["ch4"]), 1),
        caption="CH₄ (ppm)",
    )

    # Add each colormap with vertical offsets to avoid overlap
    add_colormap_to_map(m, altitude_colormap, position="bottomleft", idx=4)
    add_colormap_to_map(m, co2_colormap, position="bottomleft", idx=3)
    add_colormap_to_map(m, co_colormap, position="bottomleft", idx=2)
    add_colormap_to_map(m, ch4_colormap, position="bottomleft", idx=1)
    return m


def plot_picarro_ghg(df: pd.DataFrame) -> plt.Figure:
    """
    Plot a dual-axis time series of greenhouse gas (GHG) data from the FireAlarm query.

    The top subplot shows GPS altitude (in km) colored by CO₂ concentration.
    The bottom subplot shows water vapor concentration over time.

    Args:
        df (pd.DataFrame): DataFrame indexed by datetime containing
            columns 'gps_altitude', 'co2', and 'water_vapor'.

    Returns:
        plt.Figure: The matplotlib Figure object containing the plots.
    """
    # Create a figure with two vertically stacked subplots sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(16, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.05},
        constrained_layout=True,
    )

    # Top plot: Scatter altitude (km) colored by CO2 concentration
    co2_scatter = ax1.scatter(
        df.index,
        df["gps_altitude"] / 1000,  # convert meters to km
        c=df["co2"],
        cmap="coolwarm",
        s=6,
    )
    ax1.set_ylabel("Altitude (km)")
    ax1.set_title("CO₂ and Water Vapor by Time")

    # Add colorbar for CO2 values next to the top plot
    cbar = fig.colorbar(co2_scatter, ax=ax1, orientation="vertical", pad=0.01)
    cbar.set_label("CO₂ (ppm)")

    # Bottom plot: Scatter water vapor over time, black color
    ax2.scatter(df.index, df["water_vapor"], color="black", s=5)
    ax2.set_ylabel("Water Vapor (ppm)")
    ax2.set_xlabel("Time")

    # Format x-axis dates for better readability
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    fig.autofmt_xdate()

    return fig


def plot_picarro_ghg_correlation(df: pd.DataFrame) -> plt.Figure:
    """
    Plot a Pearson correlation heatmap for selected GHG variables and altitude.

    Computes the Pearson correlation coefficients between CO, CO2, CH4, water vapor,
    and GPS altitude, then displays them as a heatmap.

    Args:
        df (pd.DataFrame): DataFrame containing columns
            'co', 'co2', 'ch4', 'water_vapor', and 'gps_altitude'.

    Returns:
        plt.Figure: The matplotlib Figure object containing the heatmap.
    """
    # Calculate Pearson correlation matrix for selected columns
    corr_matrix = df[["co", "co2", "ch4", "water_vapor", "gps_altitude"]].corr(method="pearson")

    # Create figure for the heatmap
    fig = plt.figure(figsize=(8, 6))

    # Plot heatmap with annotations and diverging 'coolwarm' colormap
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")

    # Add title
    plt.title("Pearson Correlation Matrix")

    return fig


def madre_fire_gibs() -> Image:
    """
    Generate a GIF of GIBS satellite imagery for the Madre Fire area over a range of dates.

    This function queries NASA's GIBS WMS service for daily MODIS corrected reflectance imagery,
    draws a bounding box over the fire area, labels each frame with the date, and compiles the
    frames into a GIF saved locally as "images/Madre_Fire.gif".

    No arguments are required, and the GIF is saved to disk.
    """

    def latlon_to_pixel(lat: float, lon: float, bbox: list[float], width: int, height: int) -> tuple[int, int]:
        """
        Convert geographic coordinates (latitude, longitude) to pixel coordinates
        within an image defined by bbox and pixel dimensions.

        Args:
            lat (float): Latitude of the point.
            lon (float): Longitude of the point.
            bbox (list[float]): Bounding box as [min_lat, min_lon, max_lat, max_lon].
            width (int): Image width in pixels.
            height (int): Image height in pixels.

        Returns:
            (int, int): Pixel coordinates (x, y).
        """
        min_lat, min_lon, max_lat, max_lon = bbox
        x = int((lon - min_lon) / (max_lon - min_lon) * width)
        y = int((max_lat - lat) / (max_lat - min_lat) * height)
        return x, y

    width = 500
    height = 500

    # Define bounding box for Madre Fire region: [min_lat, min_lon, max_lat, max_lon]
    bbox = [34, -121, 35.75, -119]

    # Convert lat/lon of top-left and bottom-right corners of the bounding box to pixel coordinates
    x1, y1 = latlon_to_pixel(35.3, -120.225, bbox, width, height)  # top-left corner of fire region
    x2, y2 = latlon_to_pixel(34.95, -119.5, bbox, width, height)  # bottom-right corner of fire region

    # GIBS layers to request for imagery
    layers = [
        "BlueMarble_NextGeneration",
        "MODIS_Aqua_CorrectedReflectance_TrueColor",
        "MODIS_Terra_CorrectedReflectance_TrueColor",
        "Reference_Features",
        "Reference_Labels",
    ]

    # Load font for annotation
    font = ImageFont.truetype("Roboto-Bold.ttf", size=18)
    frames = []

    # Create date range for imagery (July 1 to July 7, 2025)
    dates = pd.date_range(datetime(2025, 7, 1), datetime(2025, 7, 8) - timedelta(days=1), freq="d")

    print(f"Obtaining {len(dates)} days of imagery from GIBS...")

    # Loop over dates to fetch and process images
    for day in dates[:10]:
        datatime = day.strftime("%Y-%m-%d")

        # Build WMS GetMap request URL with parameters
        gibs_wms_url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?"
        params = {
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetMap",
            "layers": ",".join(layers),
            "styles": "",
            "srs": "epsg:4326",  # Map projection
            "crs": "epsg:4326",
            "bbox": ",".join(map(str, bbox)),  # Bounds
            "width": width,
            "height": height,
            "time": datatime,  # Date of the imagery
            "format": "image/png",
            "transparent": "TRUE",
            "exceptions": "XML",
        }

        # Request image from GIBS
        url = f'{gibs_wms_url}{"&".join([f"{k}={v}" for k,v in params.items()])}'
        response = requests.get(url)
        image = I.open(io.BytesIO(response.content)).convert("RGBA")  # Ensure RGBA for transparency

        # Draw red bounding box around fire region
        draw = ImageDraw.Draw(image)
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)

        # Prepare centered label text with date
        label = f"MODIS Corrected Reflectance - {datatime}"
        bbox_text = draw.textbbox((0, 0), label, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        x = (image.width - text_width) // 2
        y = 12

        # Draw the label text in black
        draw.text((x, y), label, fill="black", font=font)

        # Add processed frame to list
        frames.append(image)

    os.makedirs("images", exist_ok=True)
    # Save all frames as an animated GIF
    frames[0].save(
        "images/Madre_Fire.gif", format="GIF", append_images=frames[1:], save_all=True, duration=1000, loop=0
    )
    return Image("images/Madre_Fire.gif")


def madre_fire_airnow_sites() -> pd.DataFrame:
    """
    Obtain and format AirNow monitoring site data near the Madre Fire area.

    This function queries the AirNow data provider for air quality monitoring
    sites within a specified bounding box near the Madre Fire, processes the
    JSON response into a pandas DataFrame, filters and cleans the data by removing
    duplicate platform names, filters out observation counts that are zero or less,
    and aligns units to available observations.

    Returns:
        pd.DataFrame: DataFrame of AirNow monitoring sites with relevant metadata.
    """
    # Request AirNow site statistics within specified bounding box around Madre Fire
    resp = requests.get(
        "https://ideas-digitaltwin.jpl.nasa.gov/insitu/1.0/sub_collection_statistics?"
        "provider=AirNow&project=air_quality&bbox=-120.5,34.5,-119,36",
        timeout=60,
    )

    # Parse JSON response
    airnow_collections_data = resp.json()
    airnow_collections_data = airnow_collections_data["providers"][0]["projects"][0]["platforms"]

    # Convert list of platforms to DataFrame, drop duplicate platform short names, reset index
    airnow_collections_df = (
        pd.DataFrame(airnow_collections_data).drop_duplicates(["platform_short_name"]).reset_index(drop=True)
    )

    # Filter observation_counts dict to keep only entries with count > 0
    airnow_collections_df["observation_counts"] = airnow_collections_df["observation_counts"].apply(
        lambda d: {k: v for k, v in d.items() if v > 0}
    )

    # Filter units dict to keep only units for observations that exist (count > 0)
    airnow_collections_df["units"] = airnow_collections_df.apply(
        lambda row: {k: row["units"].get(k) for k in row["observation_counts"].keys()}, axis=1
    )

    # Remove columns containing 'depth' in their name
    airnow_collections_df = airnow_collections_df[[c for c in airnow_collections_df.columns if "depth" not in c]]

    return airnow_collections_df


def map_madre_fire_airnow_sites(madre_airnow_collections_df: pd.DataFrame) -> folium.Map:
    """
    Create a folium map displaying AirNow monitoring sites near the Madre Fire.

    This function initializes a folium map centered near the Madre Fire region,
    overlays a MODIS Aqua True Color satellite imagery tile layer from NASA GIBS,
    draws a blue rectangle around the Madre Fire bounding box, and plots red circle
    markers for each AirNow monitoring site in the provided DataFrame with popups
    and tooltips showing the platform short name.

    Args:
        madre_airnow_collections_df (pd.DataFrame): DataFrame containing AirNow site metadata,
            expected to have 'lat', 'lon', and 'platform_short_name' columns.

    Returns:
        folium.Map: Folium map object with AirNow sites and satellite imagery.
    """
    # Initialize folium map centered near Madre Fire location
    m = folium.Map(location=[35, -119.75], zoom_start=8, tiles=None)

    # Add MODIS Aqua Corrected Reflectance True Color tile layer from NASA GIBS
    tile_url = (
        "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
        "MODIS_Aqua_CorrectedReflectance_TrueColor/default/2025-07-07/"
        "GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg"
    )
    folium.TileLayer(
        tiles=tile_url,
        attr="NASA GIBS",
        name="MODIS_Aqua_CorrectedReflectance_TrueColor",
        overlay=True,
        control=True,
        max_zoom=9,  # Level9 tile pyramid max zoom
        min_zoom=1,
        show=True,
    ).add_to(m)

    # Draw bounding box rectangle for Madre Fire area
    folium.Rectangle(
        bounds=[
            [34.95, -120.225],  # Southwest corner (lat, lon)
            [35.3, -119.5],  # Northeast corner (lat, lon)
        ],
        color="blue",
        weight=2,
        fill=True,
        fill_opacity=0,
        tooltip="Madre Fire",
    ).add_to(m)

    # Add AirNow monitoring sites as red circle markers with popups/tooltips
    for _, row in madre_airnow_collections_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            fill=True,
            fill_opacity=0.7,
            color="red",
            popup=folium.Popup(row["platform_short_name"]),
            tooltip=row["platform_short_name"],
        ).add_to(m)

    return m


def obtain_madre_airnow_data(platforms: List[str]) -> pd.DataFrame:
    """
    Query the FireAlarm system for Madre Fire relevant AirNow data.

    The function requests data from FireAlarm for the specified platforms over a
    defined date range and geographic bounding box. It returns the results as a
    Pandas DataFrame.

    Args:
        platforms (List[str]): List of platform names or IDs to query.

    Returns:
        pd.DataFrame: DataFrame containing AirNow data for the Madre Fire region
        and specified time range.
    """
    # Define start and end datetime objects for data query
    start_time = datetime(2025, 6, 25)
    end_time = datetime(2025, 7, 7)

    # Define bounding box as a string: "min_lon,min_lat,max_lon,max_lat"
    bounding_box = "-120.5,34.5,-119,36"

    # Query FireAlarm data using data_request
    madre_airnow_df = data_request(
        platforms,
        start_time,
        end_time,
        bounding_box,
        provider="AirNow",
    )

    return madre_airnow_df


def obtain_madre_satellite_data(collection: str) -> Tuple[List[str], List[float]]:
    """
    Query the digital twin API for satellite time series data over the Madre Fire region.

    Args:
        collection (str): The satellite data collection identifier to query.

    Returns:
        Tuple[List[str], List[float]]: Two lists containing ISO timestamp strings and
        corresponding maximum values from the data.
    """
    madre_start = datetime(2025, 6, 25)
    madre_end = datetime(2025, 7, 8)
    madre_bb = "-120.225,34.95,-119.5,35.3"

    params = {
        "ds": collection,
        "b": madre_bb,
        "startTime": madre_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endTime": madre_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    response = requests.get("https://ideas-digitaltwin.jpl.nasa.gov/nexus/timeSeriesSpark", params=params)

    timesteps = []
    values = []
    for hit in response.json()["data"]:
        timesteps.append(hit[0]["iso_time"])
        values.append(hit[0]["max"])

    return timesteps, values


def plot_madre_data(
    co_timesteps: List[str],
    co_values: List[float],
    hcho_timesteps: List[str],
    hcho_values: List[float],
    madre_airnow_df: pd.DataFrame,
) -> plt.Figure:
    """
    Plot time series data for Madre Fire: TROPOMI CO, TEMPO HCHO, and AirNow PM10 measurements.

    Args:
        co_timesteps (List[str]): ISO timestamps for Carbon Monoxide data.
        co_values (List[float]): CO values.
        hcho_timesteps (List[str]): ISO timestamps for HCHO data.
        hcho_values (List[float]): HCHO values.
        madre_airnow_df (pd.DataFrame): AirNow PM10 data with platform info and datetime index.

    Returns:
        plt.Figure: Matplotlib Figure object with the plots.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # Plot CO
    ax1.plot(pd.to_datetime(co_timesteps), co_values, label="TROPOMI Carbon Monoxide", color="tab:red")
    ax1.set_ylabel("mol m⁻²")
    ax1.legend()

    # Plot HCHO
    ax2.plot(pd.to_datetime(hcho_timesteps), hcho_values, label="TEMPO HCHO", color="tab:blue")
    ax2.set_ylabel("molecules/cm²")
    ax2.legend()

    # Plot AirNow PM10 data grouped by platform
    for platform, group in madre_airnow_df[["pm10", "platform"]].dropna(axis=0).groupby("platform")["pm10"]:
        ax3.plot(group.index, group.values, label=platform)
    ax3.set_ylabel("µg/m³")
    ax3.set_title("AirNow PM 10")
    ax3.legend(loc="upper right")

    # Format x-axis dates nicely
    fig.autofmt_xdate()
    ax1.set_title("Madre Fire")
    plt.tight_layout()
    return fig


def obtain_roze_o3_data() -> pd.DataFrame:
    """
    Query ROZE O3 data from the FireAlarm system, clean it, fit a linear regression
    model of O3 against GPS altitude, and calculate residuals.

    Returns:
        pd.DataFrame: DataFrame containing O3 data with residuals from altitude regression.
    """
    # Request data for the ROZE O3 platform in the specified bounding box and date range
    ROZE_O3_df = data_request(
        "Dynamic-Aviation-B200--ROZE-O3", datetime(2025, 6, 1), datetime(2025, 7, 3), "-80,38,-70,40"
    )

    # Replace fill values (-9999) with NaN
    ROZE_O3_df["o3"] = ROZE_O3_df["o3"].replace(-9999, np.nan)

    # Filter out rows where O3 is NaN
    roze_df = ROZE_O3_df[ROZE_O3_df["o3"].notna()]

    # Prepare feature matrix (altitude) and target vector (O3)
    X = roze_df["gps_altitude"].values.reshape(-1, 1)
    y = roze_df["o3"].values

    # Fit linear regression model of O3 vs altitude
    model = LinearRegression().fit(X, y)

    # Predict O3 based on altitude
    predicted_o3 = model.predict(X)

    # Calculate residuals: actual - predicted O3
    roze_df["o3_residual"] = y - predicted_o3

    return roze_df


def plot_roze_o3_correlation(roze_o3_df: pd.DataFrame) -> plt.Figure:
    """
    Plot a smoothed vertical profile of O3 concentration against altitude,
    including Pearson correlation coefficient and p-value in the title.

    Args:
        roze_o3_df (pd.DataFrame): DataFrame containing 'o3' and 'gps_altitude' columns.

    Returns:
        plt.Figure: Matplotlib figure object containing the plot.
    """
    # Calculate Pearson correlation and p-value between altitude and O3
    corr, p_value = pearsonr(roze_o3_df["gps_altitude"], roze_o3_df["o3"])

    fig = plt.figure(figsize=(7, 8))
    sns.regplot(
        x="o3",
        y="gps_altitude",
        data=roze_o3_df,
        lowess=True,  # Locally weighted scatterplot smoothing
        scatter_kws={"s": 10, "alpha": 0.3},
        line_kws={"color": "red"},
    )

    plt.xlabel("O3 (ppb)")
    plt.ylabel("Altitude (m)")
    plt.title(f"Smoothed Vertical Profile of ROZE O3\nPearson correlation: {corr:.3f}, p-value: {p_value:.3g}")
    plt.grid(True)
    return fig


def obtain_tempo_o3_data(roze_o3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Query TEMPO O3 satellite data averaged over a bounding box defined by roze_o3_df.

    Args:
        roze_o3_df (pd.DataFrame): DataFrame with 'latitude' and 'longitude' columns to define bounding box.

    Returns:
        pd.DataFrame: DataFrame containing latitude, longitude, and mean O3 concentration ('o3_mean').
    """
    dataset_name = "TEMPO_O3TOT_L3_V03"
    start_time = datetime(2025, 6, 22)
    end_time = datetime(2025, 6, 25)
    bounding_box = (
        f'{roze_o3_df["longitude"].min()},'
        f'{roze_o3_df["latitude"].min()},'
        f'{roze_o3_df["longitude"].max()},'
        f'{roze_o3_df["latitude"].max()}'
    )

    params = {
        "ds": dataset_name,
        "b": bounding_box,
        "startTime": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endTime": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    response = requests.get("https://ideas-digitaltwin.jpl.nasa.gov/nexus/timeAvgMapSpark", params=params)
    response_data = response.json()

    # Flatten nested data structure to list of records with lat, lon, and mean O3 value
    records = [
        {"lat": data["lat"], "lon": data["lon"], "o3_mean": data["mean"]}
        for row in response_data["data"]
        for data in row
    ]

    tempo_o3_df = pd.DataFrame(records)
    return tempo_o3_df


def map_o3_data(roze_df: pd.DataFrame, tempo_o3_df: pd.DataFrame) -> folium.Map:
    """
    Create an interactive folium map displaying:
    - ROZE O3 residuals as circle markers colored by residual value.
    - TEMPO O3 column data as colored polygons on a grid.

    Args:
        roze_df (pd.DataFrame): DataFrame with 'latitude', 'longitude', 'o3_residual', 'o3', 'gps_altitude'.
        tempo_o3_df (pd.DataFrame): DataFrame with 'lat', 'lon', and 'o3_mean' columns.

    Returns:
        folium.Map: Interactive map with the O3 data layers.
    """
    # Center map on mean coordinates from ROZE data
    center_lat = roze_df["latitude"].mean()
    center_lon = roze_df["longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8)

    # Add MODIS Aqua Corrected Reflectance True Color tile layer from NASA GIBS
    tile_url = (
        "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
        "MODIS_Aqua_CorrectedReflectance_TrueColor/default/2025-06-24/"
        "GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg"
    )
    folium.TileLayer(
        tiles=tile_url,
        attr="NASA GIBS",
        name="MODIS_Aqua_CorrectedReflectance_TrueColor",
        overlay=True,
        control=True,
        max_zoom=9,  # Level9 tile pyramid max zoom
        min_zoom=1,
        show=True,
    ).add_to(m)

    # ROZE residuals layer (circle markers)
    roze_residual_layer = folium.FeatureGroup(name="ROZE O3 residual", show=True, overlay=True, control=True)
    norm = colors.Normalize(vmin=roze_df["o3_residual"].min(), vmax=roze_df["o3_residual"].max())
    cmap = plt.cm.coolwarm  # Diverging colormap for residuals

    for _, row in roze_df.iterrows():
        color = colors.to_hex(cmap(norm(row["o3_residual"])))
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=1,
            popup=(
                f"ROZE\nresidual: {row['o3_residual']:.2f}\n"
                f"raw: {row['o3']:.2f} (ppb)\nalt: {row['gps_altitude']:.4f} (m)"
            ),
        ).add_to(roze_residual_layer)

    # TEMPO O3 column layer (polygons)
    tempo_min, tempo_max = tempo_o3_df["o3_mean"].min(), tempo_o3_df["o3_mean"].max()
    norm_tempo = colors.Normalize(vmin=tempo_min, vmax=tempo_max)
    cmap_tempo = cm.get_cmap("viridis")

    shrink_factor = 0.8  # adjust size of grid squares visually
    half_lat = 0.0125 * shrink_factor
    half_lon = 0.0125 * shrink_factor

    tempo_layer = folium.FeatureGroup(name="TEMPO O3 Column (DU)", show=True, overlay=True, control=True)

    for _, row in tempo_o3_df.iterrows():
        bounds = [
            [row["lat"] - half_lat, row["lon"] - half_lon],  # southwest
            [row["lat"] - half_lat, row["lon"] + half_lon],  # southeast
            [row["lat"] + half_lat, row["lon"] + half_lon],  # northeast
            [row["lat"] + half_lat, row["lon"] - half_lon],  # northwest
        ]
        color = colors.rgb2hex(cmap_tempo(norm_tempo(row["o3_mean"])))
        folium.Polygon(
            locations=bounds,
            color=color,
            fill=True,
            fill_opacity=0.6,
            weight=0,
            popup=f"TEMPO O3 Column: {row['o3_mean']:.2f} DU",
        ).add_to(tempo_layer)

    # Add layers to map
    tempo_layer.add_to(m)
    roze_residual_layer.add_to(m)

    o3_resid_colormap = LinearColormap(
        colors=[cm.coolwarm(i) for i in np.linspace(0, 1, 256)],
        vmin=round(np.nanmin(roze_df["o3_residual"]), 1),
        vmax=round(np.nanmax(roze_df["o3_residual"]), 1),
        caption="O3 residual",
    )

    step_edges = np.linspace(
        round(np.nanmin(roze_df["o3_residual"]), 1), round(np.nanmax(roze_df["o3_residual"]), 1), 11
    )
    o3_resid_colormap.tick_labels = {val: f"{val:.1f}" for val in step_edges}

    tempo_o3_colormap = LinearColormap(
        colors=[cm.viridis(i) for i in np.linspace(0, 1, 256)],
        vmin=round(tempo_min, 1),
        vmax=round(tempo_max, 1),
        caption="TEMPO O3 (DU)",
    )

    # Add each colormap with vertical offsets to avoid overlap
    add_colormap_to_map(m, o3_resid_colormap, position="bottomleft", idx=1)
    add_colormap_to_map(m, tempo_o3_colormap, position="bottomleft", idx=2)

    return m
