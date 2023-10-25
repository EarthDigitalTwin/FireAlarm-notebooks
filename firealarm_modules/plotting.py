import textwrap
import types
from datetime import datetime
from typing import Dict, Literal
from typing import List, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from matplotlib import dates
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap


def timeseries_plot(data: List[Tuple[xr.DataArray, str]], x_label: str, y_label: str, title='', norm=False):
    '''
    Plots timeseries data on a chart
    '''

    plt.figure(figsize=(12, 5))

    for entry in data:
        da = entry[0]
        label = entry[1]
        vals = da.values
        if norm:
            vals = (vals - np.nanmin(vals)) / (np.nanmax(vals) - np.nanmin(vals))
        if len(entry) == 3:
            plt.plot(da.time, vals, linewidth=2,
                     label=textwrap.fill(label, 50), color=entry[2])
        else:
            plt.plot(da.time, vals, linewidth=2,
                     label=textwrap.fill(label, 50))

    plt.grid(visible=True, which='major', color='k', linestyle='-')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.gcf().autofmt_xdate()
    plt.xticks(rotation=45)
    plt.title(title, fontsize=16)
    plt.legend(prop={'size': 10})
    plt.show()


def timeseries_plot_irregular(
        data: List[Tuple[xr.DataArray, str]],
        x_label: str,
        y_label: str,
        title='',
        norm=False,
        times: List[datetime] = None,
        max_ticks=12,
        save_as=None
):
    plt.figure(figsize=(12, 5))

    if times:
        data = [(da.sel(time=times, method='nearest'), l) for da, l in data]

    for entry in data:
        da = entry[0]
        label = entry[1]
        vals = da.values

        x_ticks = [str(dt)[:19] for dt in da.time.to_numpy()]

        if norm:
            vals = (vals - np.nanmin(vals)) / (np.nanmax(vals) - np.nanmin(vals))
        if len(entry) == 3:
            plt.plot(np.arange(len(vals)), vals, linewidth=2,
                     label=textwrap.fill(label, 50), color=entry[2])
        else:
            plt.plot(np.arange(len(vals)), vals, linewidth=2,
                     label=textwrap.fill(label, 50))

        if max_ticks:
            s = (len(vals) // max_ticks) + 1
        else:
            s = 1

        plt.xticks(np.arange(len(vals))[::s], x_ticks[::s])

    plt.grid(visible=True, which='major', color='k', linestyle='-')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.gcf().autofmt_xdate()
    plt.xticks(rotation=45)
    plt.title(title, fontsize=16)
    plt.legend(prop={'size': 10})

    if save_as:
        plt.savefig(save_as, facecolor='white')
    plt.show()


def timeseries_multi_plot(data: List[Tuple[xr.DataArray, str, str]]):
    
    fig, axs = plt.subplots(int(np.ceil(len(data)/2)), 2,  figsize=(10,8))
    # fig.autofmt_xdate()
    # fig.xtic
    for (da, title, ylabel), ax in zip(data, axs.flatten()):
        ax.grid(visible=True, which='major', color='k', linestyle='-')
        ax.plot(da.time, da.values)
        ax.set_ylabel(ylabel, fontsize=12)
        for xlabel in ax.get_xticklabels():
            xlabel.set_ha("right")
            xlabel.set_rotation(30)
        ax.set_title(title, fontsize=14)
        # ax.legend(prop={'size': 12})
    if len(data)%2:
        axs.flatten()[-1].set_visible(False)
    fig.tight_layout()
    plt.show()  

def timeseries_bands_plot(da, var_label, x_label, y_label, title):
    plt.figure(figsize=(12, 5))

    plt.fill_between(da['time'], da['mean'] - da['std'], da['mean'] + da['std'], alpha=.25)
    plt.plot(da['time'], da['mean'], linewidth=2, label=textwrap.fill(var_label, 50))

    plt.grid(visible=True, which='major', color='k', linestyle='-')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.gcf().autofmt_xdate()
    # plt.gca().xaxis.set_major_formatter(dates.ConciseDateFormatter(plt.gca().xaxis.get_major_locator()))
    plt.xticks(rotation=45)
    plt.title(title, fontsize=16)
    plt.legend(prop={'size': 12})
    plt.show()


def plot_insitu(data: List[Tuple[pd.DataFrame, str, str]], title: str, ylabel='m3/s', norm=False, shared_year=False):
    fig = plt.figure(figsize=(12, 5))

    for df, var, label in data:
        if var == 'Streamflow':
            var_data = df[var]/35.315
        else:
            var_data = df[var]

        if norm:
            var_data = (var_data - np.min(var_data)) / (np.max(var_data) - np.min(var_data))
            ylabel = 'Normalized values'
        plt.plot(df.time, var_data, label=label)

    plt.grid(visible=True, which='major', color='k', linestyle='-')
    plt.ylabel(ylabel, fontsize=12)
    if shared_year:
        plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%b %d'))
        
    plt.gcf().autofmt_xdate()
    plt.xticks(rotation=45)
    plt.title(title, fontsize=16)
    plt.legend(prop={'size': 12})


def map_with_timeseries_multi(
        data: List[Tuple[xr.Dataset, str, Dict[str, float], xr.DataArray]],
        padding=(2.0, 0.5),
        title='',
        var: Literal['mean', 'minimum', 'maximum'] = 'mean',
        cmap='rainbow',
        vmin=None, vmax=None,
        cbar_label='',
        ts_title='Timeseries',
        ts_xlabel='time',
        ts_ylabel='value',
        context_map_title='Map',
        subset_map_title='Data Subset',
        quick=False
):
    n_rows = len(data)
    n_cols = 3

    n_subplots = n_rows * n_cols

    pos = range(1, n_subplots + 1)

    fig = plt.figure(
        1,
        figsize=(22, 3 * n_rows),
        constrained_layout=True
    )

    gs = fig.add_gridspec(n_rows, 5, wspace=0.000)

    for i, d in enumerate(data):
        # ax1 = fig.add_subplot(n_rows, n_cols, pos[i*3])
        ax1 = fig.add_subplot(gs[i, 0])

        min_lat = d[2]['min_lat']
        min_lon = d[2]['min_lon']
        max_lat = d[2]['max_lat']
        max_lon = d[2]['max_lon']

        b_lats = [min_lat, max_lat, max_lat, min_lat]
        b_lons = [min_lon, min_lon, max_lon, max_lon]

        lat_len = max_lat - min_lat
        lon_len = max_lon - min_lon

        if lat_len >= lon_len:
            diff = lat_len - lon_len

            min_lon -= (diff / 2)
            max_lon += (diff / 2)
        else:
            diff = lon_len - lat_len

            min_lat -= (diff / 2)
            max_lat += (diff / 2)

        margin = (max_lon - min_lon) * padding[0]

        m = Basemap(
            projection='cyl',
            lon_0=180,
            llcrnrlat=min_lat - margin,
            urcrnrlat=max_lat + margin,
            llcrnrlon=min_lon - margin,
            urcrnrlon=max_lon + margin,
            ax=ax1,
        )

        ax1.set_ylabel(d[1], fontsize='large')

        x, y = m(b_lons, b_lats)
        xy = zip(x, y)

        poly = Polygon(list(xy), edgecolor='red', facecolor='none')
        ax1.add_patch(poly)

        m.arcgisimage(server='http://server.arcgisonline.com/ArcGIS',
                      service='World_Imagery',
                      xpixels=2000, ypixels=None, dpi=2000, verbose=False)

        # m.drawcoastlines()

        if len(d) == 4:
            subset_data = d[3][0]

            sub_lats = subset_data.lat
            sub_lons = subset_data.lon

            min_lat = sub_lats.min().data
            max_lat = sub_lats.max().data
            min_lon = sub_lons.min().data
            max_lon = sub_lons.max().data

            lat_len = max_lat - min_lat
            lon_len = max_lon - min_lon

            if lat_len >= lon_len:
                diff = lat_len - lon_len

                min_lon -= (diff / 2)
                max_lon += (diff / 2)
            else:
                diff = lon_len - lat_len

                min_lat -= (diff / 2)
                max_lat += (diff / 2)

            # ax2 = fig.add_subplot(n_rows, n_cols, pos[(i*3) + 1])
            ax2 = fig.add_subplot(gs[i, 1])

            margin = (max_lon - min_lon) * padding[1]

            m2 = Basemap(
                projection='cyl',
                lon_0=180,
                llcrnrlat=min_lat - margin,
                urcrnrlat=max_lat + margin,
                llcrnrlon=min_lon - margin,
                urcrnrlon=max_lon + margin,
                ax=ax2,
            )

            x, y = m2(subset_data.lon.to_numpy(), subset_data.lat.to_numpy())

            if vmax is None:
                vmax = np.nanmax(subset_data.values)
            elif isinstance(vmax, types.FunctionType):
                vmax = vmax(subset_data)
            if vmin is None:
                vmin = np.nanmin(subset_data.values)
            elif isinstance(vmin, types.FunctionType):
                vmin = vmin(subset_data)

            cs = m2.pcolormesh(x, y, subset_data.to_numpy(), vmin=vmin, vmax=vmax, cmap=cmap, alpha=0.75)
            cb = plt.colorbar(cs, ax=ax2, label=cbar_label)

            m2.arcgisimage(server='http://server.arcgisonline.com/ArcGIS',
                           service='World_Imagery',
                           xpixels=2000, ypixels=None, dpi=2000, verbose=False)

            # m2.drawcoastlines()

            ax2.set_xlabel(f'Subset plotted from\n{d[3][1].strftime("%Y-%m-%dT%H:%M:%S")}', fontsize='medium')

        if i == 0:
            ax1.set_title(context_map_title, fontsize='large')
            ax2.set_title(subset_map_title, fontsize='large')

        # ax3 = fig.add_subplot(n_rows, n_cols, pos[(i*3) + 2])
        ax3 = fig.add_subplot(gs[i, 2:])

        da = d[0][var]
        label = d[1]
        vals = da.values

        ax3.plot(da.time, vals, linewidth=2, label=textwrap.fill(label, 50))
        ax3.grid(visible=True, which='major', color='k', linestyle='-')
        if i == 0:
            ax3.set_title(ts_title, fontsize='large')
        ax3.set_xlabel(ts_xlabel, fontsize='large')
        ax3.set_ylabel(ts_ylabel, fontsize='large')

    fig.suptitle(title, fontsize='xx-large')

    plt.savefig('emit.png', facecolor='white')
    plt.show()


def base_map(bounds: dict = {}, padding: float = 2.5) -> plt.axes:
    '''
    Creates map with bounds and padding
    '''
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    if bounds:
        bounds = (bounds['min_lon'] - padding,
                  bounds['max_lon'] + padding,
                  bounds['min_lat'] - padding,
                  bounds['max_lat'] + padding)
    else:
        bounds = (-180, 180, -90, 90)
    ax.set_extent(bounds, ccrs.PlateCarree())

    ax.add_feature(cf.LAND)
    ax.add_feature(cf.OCEAN)
    ax.coastlines('10m')
    ax.add_feature(cf.STATES, zorder=100)
    roads = cf.NaturalEarthFeature(category='cultural', 
        name='roads',
        scale='10m',facecolor='none')
    ax.add_feature(roads, alpha=.5)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black',
                      alpha=0.25, linestyle='--', draw_labels=True, zorder=90)
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right = False
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return ax


def map_box(bb: dict, points: List= [], padding=20):
    '''
    Adds bounding box to map
    '''
    ax = base_map(bb, padding)
    poly = Polygon([(bb['min_lon'], bb['min_lat']), (bb['min_lon'], bb['max_lat']), (bb['max_lon'], bb['max_lat']), (bb['max_lon'], bb['min_lat'])],
                   facecolor=(0, 0, 0, 0.0), edgecolor='red', linewidth=2, zorder=200)
    ax.add_patch(poly)
    for (lat, lon, label) in points:
        ax.scatter([lon], [lat], s=50, alpha=1, label=label)
    if points:
        plt.legend(facecolor='white', framealpha=1)
    plt.show()


def map_points(points: List, region='', title='', zoom=False):
    '''
    Plots lat lon points on map
    points: list of tuples (lat, lon, label)
    '''
    ax = base_map()
    for (lat, lon, label) in points:
        if "Fire" in label:
            ax.scatter([lon], [lat], s=100, marker='*', alpha=1, label=label)
        else:
            ax.scatter([lon], [lat], s=100, alpha=1, label=label)

    ax.set_title(title)

    if region == 'miss':
        ax.set_xlim(-95, -86)
        ax.set_ylim(29, 35)

        if zoom:
            ax.set_xlim(-91.25, -90.75)
            ax.set_ylim(32.1, 32.75)
    elif region == 'gar':
        ax.set_xlim(-3, 5)
        ax.set_ylim(41, 48)

        if zoom:
            ax.set_xlim(-0.5, 1)
            ax.set_ylim(44, 45)
    elif region == 'la':
        ax.set_xlim(-120, -117)
        ax.set_ylim(32, 35)
    elif region == 'alberta':
        ax.set_xlim(-130, -110)
        ax.set_ylim(40, 65)
    elif region == 'norcal':
        ax.set_xlim(-127.5, -115)
        ax.set_ylim(35, 45)
    elif region == 'westcoast':
        ax.set_xlim(-130, -100)
        ax.set_ylim(30, 60)
    elif region == 'newyork':
        ax.set_xlim(-80, -72)
        ax.set_ylim(40,45)

    ax.legend().set_zorder(102)


def map_data(
        data: xr.DataArray,
        title: str,
        cmap='rainbow',
        cb_label='',
        log_scale=False,
        padding=2.5,
        vmin=None,
        vmax=None,
        points: List= []
):
    '''
    Plots data on map
    '''
    bounds = {
        'min_lon': data.lon.min(),
        'max_lon': data.lon.max(),
        'min_lat': data.lat.min(),
        'max_lat': data.lat.max()
    }
    ax = base_map(bounds, padding)
    x, y = np.meshgrid(data.lon, data.lat)
    if vmax is None:
        vmax = np.nanmax(data.values)
    if vmin is None:
        vmin = np.nanmin(data.values)

    if log_scale:
        mesh = ax.pcolormesh(x, y, data.values, norm=colors.LogNorm(), cmap=cmap, alpha=0.75)
    else:
        mesh = ax.pcolormesh(x, y, data.values, vmin=vmin,
                             vmax=vmax, cmap=cmap, alpha=0.75)

    for (lat, lon, label) in points:
        ax.scatter([lon], [lat], s=50, alpha=1, label=label)
    if points:
        plt.legend(facecolor='white', framealpha=1)
    cb = plt.colorbar(mesh)
    cb.set_label(cb_label)
    plt.title(title)
    plt.show()


def heatmap(data: xr.DataArray, x_label: str, y_label: str, title='', cmap='rainbow'):
    '''
    Plots colormesh heatmap
    '''
    time = [np.datetime_as_string(t, unit='D') for t in data.time]

    plt.figure(figsize=(12, 5))
    mesh = plt.pcolormesh(time, data.dim, data, cmap=cmap)
    plt.colorbar(mesh)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    locator = dates.DayLocator(interval=5)
    plt.gca().xaxis.set_major_locator(locator)
    plt.gcf().autofmt_xdate()
    plt.xticks(rotation=45)
    plt.title(title)
    plt.show()


def stacked_overlay_plot(x_datas: List[np.array], y_datas: List[np.array],
                         series_labels: List[str], y_labels=List[str], title: str = '',
                         top_paddings: List[int] = [0, 0]):

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 7))

    # Plot 1
    ax[0].set_title(title)
    ax[0].plot(
        [datetime.strptime(x_val, '%Y-%m-%dT%H:%M:%SZ').replace(year=2022) for x_val in x_datas[0]],
        y_datas[0], label=series_labels[0])

    # Plot 2
    ax[0].plot(
        [datetime.strptime(x_val, '%Y-%m-%dT%H:%M:%SZ').replace(year=2022) for x_val in x_datas[1]],
        y_datas[1], label=series_labels[1])

    ax[0].legend(loc='upper center', shadow=True)
    y_data_max = max(np.amax(y_datas[0]), np.amax(y_datas[1]))
    ax[0].set_ylim([0, y_data_max + top_paddings[0]])
    ax[0].set_ylabel(y_labels[0])

    # Plot 3
    ax[1].plot(
        [datetime.strptime(x_val, '%Y-%m-%dT%H:%M:%SZ').replace(year=2022) for x_val in x_datas[2]],
        y_datas[2], label=series_labels[2])

    # Plot 4
    ax[1].plot(
        [datetime.strptime(x_val, '%Y-%m-%dT%H:%M:%SZ').replace(year=2022) for x_val in x_datas[3]],
        y_datas[3], label=series_labels[3])

    ax[1].legend(loc='upper center', shadow=True)
    y_data_max = max(np.amax(y_datas[2]), np.amax(y_datas[3]))
    ax[1].set_ylim([0, y_data_max + top_paddings[1]])
    ax[1].set_ylabel(y_labels[1])

    # Set title and legend
    plt.legend(loc='upper center', shadow=True)

    # Set grid and ticks
    dtFmt = dates.DateFormatter('%b %d')
    plt.gca().xaxis.set_major_formatter(dtFmt)
    plt.xticks(rotation=45)
    ax[0].tick_params(left=False, bottom=False)
    ax[1].tick_params(left=False, bottom=False)
    ax[0].grid(b=True, which='major', color='k', linestyle='--', linewidth=0.25)
    ax[1].grid(b=True, which='major', color='k', linestyle='--', linewidth=0.25)

    plt.show()
