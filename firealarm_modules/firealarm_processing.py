from collections import defaultdict
from datetime import datetime
import os
from typing import List
import numpy as np
import xarray as xr
import pandas as pd
import time
import requests
from requests.compat import urlencode

DT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
BASE_URL = 'https://ideas-digitaltwin.jpl.nasa.gov'
NEXUS_URL = os.path.join(BASE_URL, 'nexus')
INSITU_URL = os.path.join(BASE_URL, 'insitu', '1.0')

'''
FireAlarm endpoint functions
'''

def firealarm_request(url):
    try:
        r = requests.get(url, verify=False)
        r.raise_for_status()
    except:
        raise Exception(f'Error processing request. Check parameters.')
    results = r.json()
    if 'data' in results or 'data' in results[0]:
        return results
    else:
        raise Exception('No data found for request.')

def spatial_timeseries(dataset: str, bb: dict, start_time: datetime, end_time: datetime) -> xr.Dataset:
    '''
    Makes request to timeSeriesSpark FireAlarm endpoint
    '''
    algo_url = os.path.join(NEXUS_URL, 'timeSeriesSpark')
    
    params = {
        'ds': dataset,
        'minLon': bb['min_lon'],
        'minLat': bb['min_lat'],
        'maxLon': bb['max_lon'],
        'maxLat': bb['max_lat'],
        'startTime': start_time.strftime(DT_FORMAT),
        'endTime': end_time.strftime(DT_FORMAT),
        'lowPassFilter': False
    }
    
    url = f'{algo_url}?{urlencode(params)}'

    # Display some information about the job
    print(url)
    print()

    # Query FireAlarm to compute the time averaged map
    print("Waiting for response from FireAlarm...", end="")
    start = time.perf_counter()
    resp_json = firealarm_request(url)
    print(f"took {time.perf_counter() - start} seconds")
    return prep_ts(resp_json)


def temporal_variance(dataset: str, bb: dict, start_time: datetime, end_time: datetime) -> xr.DataArray:
    '''
    Makes request to varianceSpark FireAlarm endpoint
    '''
    algo_url = os.path.join(NEXUS_URL, 'varianceSpark')
    
    params = {
        'ds': dataset,
        'minLon': bb['min_lon'],
        'minLat': bb['min_lat'],
        'maxLon': bb['max_lon'],
        'maxLat': bb['max_lat'],
        'startTime': start_time.strftime(DT_FORMAT),
        'endTime': end_time.strftime(DT_FORMAT)
    }
    
    url = f'{algo_url}?{urlencode(params)}'

    # Display some information about the job
    print('url\n', url)
    print()

    # Query FireAlarm to compute the time averaged map
    print("Waiting for response from FireAlarm... ", end="")
    start = time.perf_counter()
    resp_json = firealarm_request(url)
    print(f"took {time.perf_counter() - start} seconds")
    return prep_var(resp_json)


def data_subsetting(dataset: str, bb: dict, start_time: datetime, end_time: datetime, variable_name: str | None = None) -> xr.DataArray:
    '''
    Makes request to datainbounds FireAlarm endpoint
    '''
    algo_url = os.path.join(NEXUS_URL, 'datainbounds')
    
    params = {
        'ds': dataset,
        'b': f'{bb["min_lon"]},{bb["min_lat"]},{bb["max_lon"]},{bb["max_lat"]}',
        'startTime': start_time.strftime(DT_FORMAT),
        'endTime': end_time.strftime(DT_FORMAT),
    }
    
    url = f'{algo_url}?{urlencode(params)}'

    print(url)
    print()

    print("Waiting for response from FireAlarm...", end="")
    start = time.perf_counter()
    resp_json = firealarm_request(url)
    print(f"took {time.perf_counter() - start} seconds")
    return prep_data_in_bounds(resp_json, variable_name)


def max_min_map_spark(dataset: str, bb: dict, start_time: datetime, end_time: datetime) -> xr.Dataset:
    '''
    Makes request to maxMinMapSpark endpoint
    '''
    algo_url = os.path.join(NEXUS_URL, 'maxMinMapSpark')
    
    params = {
        'ds': dataset,
        'minLon': bb['min_lon'],
        'minLat': bb['min_lat'],
        'maxLon': bb['max_lon'],
        'maxLat': bb['max_lat'],
        'startTime': start_time.strftime(DT_FORMAT),
        'endTime': end_time.strftime(DT_FORMAT)
    }
    
    url = f'{algo_url}?{urlencode(params)}'

    print(url)
    print()

    print("Waiting for response from FireAlarm... ", end="")
    start = time.perf_counter()
    resp_json = firealarm_request(url)
    print("took {} seconds".format(time.perf_counter() - start))
    return max_min_prep(resp_json)


def daily_diff(dataset: str, clim: str, bb: dict, start_time: datetime, end_time: datetime) -> xr.Dataset:
    '''
    Makes request to dailydifferenceaverage_spark endpoint
    '''
    algo_url = os.path.join(NEXUS_URL, 'dailydifferenceaverage_spark')
    
    params = {
        'dataset': dataset,
        'climatology': clim,
        'b': f'{bb["min_lon"]},{bb["min_lat"]},{bb["max_lon"]},{bb["max_lat"]}',
        'startTime': start_time.strftime(DT_FORMAT),
        'endTime': end_time.strftime(DT_FORMAT),
    }
    
    url = f'{algo_url}?{urlencode(params)}'

    print(url)
    print()

    print("Waiting for response from FireAlarm... ", end="")
    start = time.perf_counter()
    resp_json = firealarm_request(url)
    print(f"took {time.perf_counter() - start} seconds")
    return daily_diff_prep(resp_json)


def temporal_mean(dataset: str, bb: dict, start_time: datetime, end_time: datetime) -> xr.DataArray:
    '''
    Makes request to timeAvgMapSpark endpoint
    '''
    algo_url = os.path.join(NEXUS_URL, 'timeAvgMapSpark')
    
    params = {
        'ds': dataset,
        'b': f'{bb["min_lon"]},{bb["min_lat"]},{bb["max_lon"]},{bb["max_lat"]}',
        'startTime': start_time.strftime(DT_FORMAT),
        'endTime': end_time.strftime(DT_FORMAT),
    }
    
    url = f'{algo_url}?{urlencode(params)}'
    
    print(url)
    print()

    print("Waiting for response from FireAlarm... ", end="")
    start = time.perf_counter()
    resp_json = firealarm_request(url)
    print(f"took {time.perf_counter() - start} seconds")
    return temporal_mean_prep(resp_json)


def hofmoeller(dataset: str, bb: dict, start_time: datetime, end_time: datetime, dim: str = 'latitude') -> xr.Dataset:
    '''
    Makes request to either latitudeTimeHofMoellerSpark or longitudeTimeHofMoellerSpark endpoint
    '''
    algo_url = os.path.join(NEXUS_URL, f'{dim}TimeHofMoellerSpark')
    
    params = {
        'ds': dataset,
        'b': f'{bb["min_lon"]},{bb["min_lat"]},{bb["max_lon"]},{bb["max_lat"]}',
        'startTime': start_time.strftime(DT_FORMAT),
        'endTime': end_time.strftime(DT_FORMAT),
    }
    
    url = f'{algo_url}?{urlencode(params)}'
    
    print(url)
    print()

    print("Waiting for response from FireAlarm... ", end="")
    start = time.perf_counter()
    resp_json = firealarm_request(url)
    print(f"took {time.perf_counter() - start} seconds")
    return hofmoeller_prep(resp_json, dim)


def insitu(provider: str, project: str, bb: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
    results = []
    next_url = f'{INSITU_URL}/query_data_doms_custom_pagination?startIndex=0&itemsPerPage=10000&' \
        f'provider={provider}&project={project}&startTime={datetime.strftime(start_time, "%Y-%m-%dT%H:%M:%SZ")}&' \
        f'endTime={datetime.strftime(end_time, "%Y-%m-%dT%H:%M:%SZ")}&bbox={bb}'

    while next_url and next_url != 'NA':
        print(next_url)
        res = requests.get(next_url)
        results.append(res.json())
        if 'next' in res.json().keys() and res.json()['next'] != next_url:
            next_url = res.json()['next']
        else:
            break
    if not results:
        raise ValueError("No data found for this query")
    return prep_insitu(results)


def get_datasets() -> pd.DataFrame:
    r = requests.get(f'{BASE_URL}/edge/ws/dat/dataset?inDAT=true&itemsPerPage=100')
    aq_datasets = pd.DataFrame([ds for ds in r.json()['Datasets'] if 'air quality' in ds['Keyword']])
    return aq_datasets

def get_date_coverage(collection: str) -> tuple[datetime, datetime]:
    r = requests.get(os.path.join(NEXUS_URL, 'list'))
    collections = r.json()
    df = pd.DataFrame(collections)
    collection_meta = df[df['shortName'] == collection]
    if collection_meta.shape[0] != 1:
        print(f'Collection {collection} not found')
        return
    start_date = pd.to_datetime(collection_meta['iso_start']).iloc[0].date()
    end_date = pd.to_datetime(collection_meta['iso_end']).iloc[0].date()
    return start_date, end_date

def get_insitu_collections() -> pd.DataFrame:
    r = requests.get(f'{INSITU_URL}/query_collection_list')
    aq_collections = []
    for collection in r.json():
        if any(project in collection['project'] for project in ['AQ', 'air_quality']):
            aq_collections.append(collection)
    return pd.DataFrame(aq_collections)[['provider', 'project']]

def get_insitu_sites(project: str, provider: str) -> pd.DataFrame:
    params = {'project': project, 'provider': provider}
    r = requests.get(f'{INSITU_URL}/sub_collection_statistics', params=params)
    sites = [site for site in r.json()["providers"][0]["projects"][0]["platforms"]]
    return pd.DataFrame(sites)[['platform', 'platform_short_name', 'lat', 'lon', 'min_datetime', 'max_datetime']]

'''
FireAlarm endpoint response processing
'''


def prep_insitu(results: List) -> pd.DataFrame:
    all_results = []
    for r in results:
        if 'results' in r.keys():
            all_results.extend(r['results'])
    df = pd.DataFrame(all_results)
    df = df.dropna(axis=1, how='all')
    df.time = pd.to_datetime(df.time)
    df = pd.concat([df.drop(['platform'], axis=1), df['platform'].apply(pd.Series)], axis=1)
    df = df.sort_values(by=['id', 'time'])
    return df


def prep_ts(ts_json: dict) -> xr.Dataset:
    '''
    Formats timeseriesspark response into xarray dataset object
    '''
    time = np.array([np.datetime64(ts[0]["iso_time"][:19])
                    for ts in ts_json["data"]])
    means = np.array([ts[0]["mean"] for ts in ts_json["data"]])
    mins = np.array([ts[0]["min"] for ts in ts_json["data"]])
    maxs = np.array([ts[0]["max"] for ts in ts_json["data"]])

    mean_da = xr.DataArray(means, coords=[time], dims=['time'], name='mean')
    min_da = xr.DataArray(mins, coords=[time], dims=['time'], name='minimum')
    max_da = xr.DataArray(maxs, coords=[time], dims=['time'], name='maximum')
    ds = xr.merge([mean_da, min_da, max_da])

    return ds


def prep_var(var_json: dict) -> xr.DataArray:
    '''
    Formats variancespark response into xarray dataarray object
    '''
    shortname = var_json['meta']['shortName']

    vals = np.array([v['variance'] for var in var_json['data'] for v in var])
    lats = np.array([var[0]['lat'] for var in var_json['data']])
    lons = np.array([v['lon'] for v in var_json['data'][0]])

    vals[vals == -9999] = np.nan

    vals_2d = np.reshape(
        vals, (len(var_json['data']), len(var_json['data'][0])))

    da = xr.DataArray(
        vals_2d, coords={"lat": lats, "lon": lons}, dims=["lat", "lon"])
    da.attrs['shortname'] = shortname
    da.attrs['units'] = '$m^2/s^2$'
    return da


def prep_data_in_bounds(var_json: dict, variable_name: str | None) -> xr.DataArray:
    '''
    Formats datainbounds response into xarray dataarray object
    '''
    lats = np.unique([o['latitude'] for o in var_json])
    lons = np.unique([o['longitude'] for o in var_json])
    times = np.unique([datetime.utcfromtimestamp(o['time']) for o in var_json])

    vals_3d = np.full((len(times), len(lats), len(lons)), np.nan)

    def extract_variable(data, variable_name):
        if variable_name is None:
            first: dict = data[0]
            return list(first.values())[0]

        variables = {}

        for v in data:
            for name in v:
                variables[name] = v[name]

        return variables.get(variable_name, np.nan)

    if len(var_json) > 0 and isinstance(var_json[0]['data'], dict):
        # Detect new DiB output structure and use that
        data_dict = {(datetime.utcfromtimestamp(data['time']), data['latitude'], data['longitude']): extract_variable(data['data']['variables'], variable_name) for data in var_json}
    else:
        data_dict = {(datetime.utcfromtimestamp(data['time']), data['latitude'], data['longitude']): data['data'][0]['variable'] for data in var_json}

    for i, t in enumerate(times):
        for j, lat in enumerate(lats):
            for k, lon in enumerate(lons):
                vals_3d[i, j, k] = data_dict.get((t, lat, lon), np.nan)

    da = xr.DataArray(
        data=vals_3d,
        dims=['time', 'lat', 'lon'],
        coords=dict(
            time=(['time'], times),
            lat=(['lat'], lats),
            lon=(['lon'], lons)
        )
    )

    return da


def max_min_prep(var_json: dict) -> xr.Dataset:
    '''
    Formats maxmin response into xarray dataset object
    '''
    shortname = var_json['meta']['shortName']
    maxima = np.array([v['maxima'] for var in var_json['data'] for v in var if v['maxima']])
    minima = np.array([v['minima'] for var in var_json['data'] for v in var])
    lat = np.array([var[0]['lat'] for var in var_json['data']])
    lon = np.array([v['lon'] for v in var_json['data'][0]])
    
    maxima = np.where(maxima==-9999.0, np.nan, maxima)
    minima = np.where(minima==-9999.0, np.nan, minima)
    

    maxima_2d = np.reshape(maxima, (len(var_json['data']), len(var_json['data'][0])))
    minima_2d = np.reshape(minima, (len(var_json['data']), len(var_json['data'][0])))

    ds = xr.Dataset(
        data_vars=dict(
            maxima=(['lat', 'lon'], maxima_2d),
            minima=(['lat', 'lon'], minima_2d)
        ),
        coords=dict(
            lat=('lat', lat),
            lon=('lon', lon)
        ),
        attrs=dict(
            shortname=shortname
        )
    )

    return ds


def daily_diff_prep(var_json: dict) -> xr.Dataset:
    '''
    Formats dailydifference response into xarray dataset object
    '''
    shortname = var_json['meta']['shortName']
    mean = np.array([v['mean'] for var in var_json['data'] for v in var])
    std = np.array([v['std'] for var in var_json['data'] for v in var])
    time = np.array([np.datetime64(v["time"], 's')
                    for var in var_json['data'] for v in var])

    ds = xr.Dataset(
        data_vars=dict(
            mean=('time', mean),
            std=('time', std)
        ),
        coords=dict(
            time=('time', time)
        ),
        attrs=dict(
            shortname=shortname
        )
    )

    return ds


def temporal_mean_prep(var_json: dict) -> xr.DataArray:
    '''
    Formats timeavgmap response into xarray dataarray object
    '''
    lat = []
    lon = []

    for row in var_json['data']:
        for data in row:
            if data['lat'] not in lat:
                lat.append(data['lat'])
            if data['lon'] not in lon:
                lon.append(data['lon'])

    lat.sort()
    lon.sort()

    da = xr.DataArray(
        data=np.zeros((len(lat), len(lon))),
        dims=['lat', 'lon'],
        coords=dict(
            lat=(['lat'], lat),
            lon=(['lon'], lon)
        )
    )

    for row in var_json['data']:
        for data in row:
            da.loc[data['lat'], data['lon']] = data['mean']
    da = da.where(da != -9999, np.nan)
    return da


def hofmoeller_prep(var_json: dict, dim: str) -> xr.Dataset:
    '''
    Formats hofmoeller response into xarray dataset object
    '''
    times = [np.datetime64(s['time'], 's') for s in var_json['data']]
    if dim == 'latitude':
        dim_short = 'lats'
    else:
        dim_short = 'lons'
    unique_dims = sorted(list(set([l[dim] for s in var_json['data'] for l in s[dim_short]])))
    
    means = defaultdict(list)
    stds = defaultdict(list)
    maxs = defaultdict(list)
    mins = defaultdict(list)
    for s in var_json['data']:
        seen_dims = []
        for l in s[dim_short]:
            seen_dims.append(l[dim])
            means[l[dim]].append(l['mean'])
            stds[l[dim]].append(l['std'])
            maxs[l[dim]].append(l['max'])
            mins[l[dim]].append(l['min'])
        unseen_dims = list(set(unique_dims) - set(seen_dims))
        for unseen_dim in unseen_dims:
            means[unseen_dim].append(np.nan)
            stds[unseen_dim].append(np.nan)
            maxs[unseen_dim].append(np.nan)
            mins[unseen_dim].append(np.nan)

    mean_2d = pd.DataFrame(means).to_numpy()
    std_2d = pd.DataFrame(stds).to_numpy()
    max_2d = pd.DataFrame(maxs).to_numpy()
    min_2d = pd.DataFrame(mins).to_numpy()

    ds = xr.Dataset(
        data_vars=dict(
            mean=(['time', dim_short[:-1]], mean_2d),
            std=(['time', dim_short[:-1]], std_2d),
            max=(['time', dim_short[:-1]], max_2d),
            min=(['time', dim_short[:-1]], min_2d)
        ),
        coords=dict(
            time=(['time'], times),
            dim=([dim_short[:-1]], unique_dims)
        )
    )
    return ds
