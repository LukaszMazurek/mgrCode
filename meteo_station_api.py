import pandas as pd
import requests
from city_cords import city_cords


if __name__ == "__main__":
    df = pd.read_csv('https://danepubliczne.imgw.pl/api/data/synop/format/csv')

    date_time = f"{df['data_pomiaru'][0]}-{df['godzina_pomiaru'][0]}"
    print(f"Pomiar z {date_time}")

    df_with_cords = []
    for _, row in df.iterrows():
        lat_lon = city_cords[row['stacja']]
        df_with_cords.append({
            'station': row['stacja'],
            'lat': lat_lon['lat'],
            'lon': lat_lon['lon'],
            'temp': row['temperatura']
        })

    df = pd.DataFrame(df_with_cords)
    q = df["temp"].quantile(0.02)
    df = df[df["temp"] > q]

    df.to_csv(f"./data/polish_data_{date_time}.csv", index=False)
