import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import math


IDX_ECCENTRIC_ANGLE = 0
IDX_EPICYCLE_ANGLE = 1
IDX_PLANET_ANGLE = 2
IDX_ECCENTRICITY = 3
IDX_RADII = 4
IDX_PE_AV = 5
IDX_ED_AV = 6
PLANETS = ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"]
COMPUTED = [
    2.251063731719305,
    1.00313819000623,
    4.725928844472276,
    0.10310843081435339,
    0.6630390873658489,
    0.01723967495680901,
    0.009174051062618753,
]
OBLIQUITY = math.radians(23.44)


def to_celestial(long: float):
    # assumes ecliptic longitude is 0
    ra = math.atan2(math.sin(long)*math.cos(OBLIQUITY), math.cos(long))
    dec = math.asin(math.sin(OBLIQUITY)*math.sin(math.radians(long)))
    return math.degrees(ra), math.degrees(dec)


def predict(props: list[float], start_time: datetime, dt: datetime):
    # get current epicycle position
    td = dt - start_time
    days = td.days + td.seconds / 86400
    if type(props[IDX_ED_AV]) is tuple:
        print(props)
    curr_ec_angle = props[IDX_EPICYCLE_ANGLE] + days * props[IDX_ED_AV]
    deferent_center = (
        props[IDX_ECCENTRICITY] * math.cos(props[IDX_ECCENTRIC_ANGLE]),
        props[IDX_ECCENTRICITY] * math.sin(props[IDX_ECCENTRIC_ANGLE])
    )
    ec_pos = (
        deferent_center[0] + math.cos(curr_ec_angle),
        deferent_center[1] + math.sin(curr_ec_angle)
    )

    # get current planet position
    curr_pl_angle = props[IDX_PLANET_ANGLE] + days * props[IDX_PE_AV]
    pl_pos = (
        ec_pos[0] + props[IDX_RADII] * math.cos(curr_pl_angle),
        ec_pos[1] + props[IDX_RADII] * math.sin(curr_pl_angle)
    )

    # atan2 to get longitude
    return ec_pos, pl_pos, math.degrees(math.atan2(pl_pos[1], pl_pos[0])) % 360


def get_pos(dt: datetime):
    start_time = datetime(1801, 1, 1)
    return to_celestial(predict(COMPUTED, start_time, dt)[2])


def get_brahe():
    csv_data = pd.read_csv(os.path.join("observations", "brahe.csv"))
    brahe_declinations = csv_data["Declination"].to_numpy()
    brahe_datetimes = [datetime(csv_data.iloc[i]["Year"].astype(np.int64), csv_data.iloc[i]["Month"].astype(np.int64), csv_data.iloc[i]["Day"].astype(np.int64)) for i in range(len(csv_data))]
    return brahe_datetimes, brahe_declinations


def get_timesteps(start: datetime, end: datetime, num_timestamps=500):
    times = [start]
    step = (end - start) / (num_timestamps - 1)
    for i in range(num_timestamps - 1):
        times.append(times[-1] + step)
    return times


def get_real(brahe_datetimes):
    real_datetimes = get_timesteps(brahe_datetimes[0], brahe_datetimes[-1])
    real_declinations = [get_pos(dt)[1] for dt in real_datetimes]
    return real_datetimes, real_declinations


def main():
    brahe_datetimes, brahe_declinations = get_brahe()
    real_datetimes, real_declinations = get_real(brahe_datetimes)
    # print(real_datetimes, real_declinations)
    plt.figure(figsize=(10, 6))
    plt.scatter(brahe_datetimes, brahe_declinations, label="brahe", c="g", s=10)
    plt.plot(real_datetimes, real_declinations, label="expected", c="b")
    plt.legend()
    plt.title("Tycho Brahe's Observations")
    plt.xlabel("Time")
    plt.ylabel("Declination")
    plt.show()


if __name__ == "__main__":
    main()