from astropy.time import Time
from astropy.coordinates import get_body_barycentric
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


def get_celestial(oc, ec):
    '''
    Returns right ascension, declination, and distance of object
    in space given it and Earth's Cartesian coords relative to the Sun.
    '''
    diff = oc-ec
    r = np.linalg.norm(diff)
    ra = np.degrees(np.arctan2(diff[1], diff[0])) % 360
    dec = np.degrees(np.arcsin(diff[2] / r))
    return ra, dec, r


def get_pos(name: str, timestamp: str):
    t = Time(timestamp)
    cartrep = get_body_barycentric(name, t)
    jc = cartrep.get_xyz().to_value()
    cartrep = get_body_barycentric('earth', t)
    ec = cartrep.get_xyz().to_value()
    return get_celestial(jc, ec)


def get_brahe():
    csv_data = pd.read_csv("brahe.csv")
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
    real_declinations = [get_pos("Mars", dt.strftime("%Y-%m-%d %H:%M:%S"))[1] for dt in real_datetimes]
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