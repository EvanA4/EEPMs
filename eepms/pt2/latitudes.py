import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt


PLANETS = ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"]
for planet in PLANETS:
    df = pd.read_csv(os.path.join("csvs", "expected", f"{planet}.csv"))
    datetimes = [datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") for timestamp in df["Timestamp"].to_list()]
    latitudes = df["Latitude"].to_list()

    plt.title(f"Latitudes of {planet[0].upper() + planet[1:]}")
    plt.plot(datetimes, latitudes)
    plt.xlabel("Time")
    plt.ylabel("Ecliptic Latitude")
    plt.show()