import time

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import folium
import pymap3d
import pymap3d.vincenty
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import gstools as gs
from kriging import Kriging

import skgstat as skg
from skgstat import Variogram



def get_bound_map(df):
    # min_lat = df.lat.min() - 0.15
    min_lat = df.lat.min() - 0.3
    # max_lat = df.lat.max() + 0.1
    max_lat = df.lat.max() + 0.1
    min_lon = df.lon.min() - 0.15
    max_lon = df.lon.max() + 0.5
    # max_lon = df.lon.max() + 1.0
    return ((min_lat, min_lon), (max_lat, max_lon))


def create_station_map(fpd_st, bound_map):
    mapf = folium.Map(location=(45, 16), zoom_start=5)
    for _, station in fpd_st.iterrows():
        marker = folium.Marker(location=(station.lat, station.lon),
                               weight=1, color="Green",
                               tooltip=f"{station.station}")
        marker.add_to(mapf)

    rect = folium.Rectangle(bound_map, color="Gray",
                            tooltip="Bounds for the temperature map to come")
    rect.add_to(mapf)

    mapf.save('polish_map.html', 'wb')


def prepare_grid(bound_map, n_width_1=50, n_width_2=50, save=False):
    lat_map_uni = np.linspace(bound_map[0][0], bound_map[1][0], n_width_1)
    print(len(lat_map_uni))
    lon_map_uni = np.linspace(bound_map[0][1], bound_map[1][1], n_width_2)

    grid = []
    for x in lat_map_uni:
        for y in lon_map_uni:
            grid.append((x, y))

    if save:
        filename = f"data/grid_{n_width_1}x{n_width_2}.csv"
        df = pd.DataFrame(grid, columns=['lat', 'lon'])
        df.index.name = 'idx'
        df.to_csv(filename, index=True)

    return grid


if __name__ == '__main__':
    df = pd.read_csv('./data/polish_data_2023-03-04-17.csv')
    bound_map = get_bound_map(df)
    print(bound_map)
    create_station_map(df, bound_map)

    lat_values = df['lat'].values
    lon_values = df['lon'].values
    temp = df['temp'].values

    cords = df[['lat', 'lon']].values

    V = skg.Variogram(coordinates=cords, values=temp, n_lags=10, model='spherical')
    print(V.describe())
    fitted_model = Variogram.fitted_model_function(**V.describe())

    kriging = Kriging(1, 1)

    n_width_1 = 50
    n_width_2 = 50

    grid = prepare_grid(bound_map, n_width_1, n_width_2, save=False)

    lat_map_uni = np.linspace(bound_map[0][0], bound_map[1][0], n_width_1)
    print(len(lat_map_uni))
    lon_map_uni = np.linspace(bound_map[0][1], bound_map[1][1], n_width_2)


    interpolation = pd.read_csv('./data/50x50_inter_spark.csv').sort_values(by=['idx'])['altitude'].values

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 12}

    matplotlib.rc('font', **font)

    temp_sk_min = df.temp.min()
    temp_sk_max = df.temp.max()

    map_temp_sk = interpolation.reshape((n_width_1, n_width_2))


    map_type = "pcolor"  # map style of kriging (expectation)

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())


    im = ax.pcolormesh(lon_map_uni, lat_map_uni, map_temp_sk,
                           vmin=temp_sk_min, vmax=temp_sk_max, cmap="jet")

    ax.scatter(lon_values, lat_values, c=df.temp.values,
               cmap="jet", edgecolor="black",
               vmin=im.get_clim()[0], vmax=im.get_clim()[1], s=100)
    plt.colorbar(im)
    ax.margins(0)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle='-', linewidth=2)
    ax.coastlines()
    ax.set_title("Kriging zwyczajny - temperatura [°C]")
    ax.set_aspect("auto")
    plt.show()

    fig.savefig('./50x50_2.jpg')
