from astropy.time import Time
from astropy.coordinates import get_body_barycentric
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os


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
    

def to_hms(ra):
    total_hours = ra / 15.0
    hours = int(total_hours)
    minutes = int((total_hours - hours) * 60)
    seconds = (total_hours - hours - minutes/60) * 3600
    return hours, minutes, seconds


def to_dsa(dec):
    deg = int(dec)
    minutes = int((dec - deg) * 60)
    seconds = (dec - deg - minutes/60) * 3600
    return deg, minutes, seconds


def get_pos(name: str, timestamp: str):
    t = Time(timestamp)
    cartrep = get_body_barycentric(name, t)
    jc = cartrep.get_xyz().to_value()
    cartrep = get_body_barycentric('earth', t)
    ec = cartrep.get_xyz().to_value()
    return get_celestial(jc, ec)


def get_timestamps(period: float, num_timestamps=10):
    timestamps = []
    dt = datetime.datetime(year=1801, month=1, day=1)    
    timestamps.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
    step_days = period / (num_timestamps - 1)
    for i in range(num_timestamps - 1):
        dt += datetime.timedelta(days=step_days)
        timestamps.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
    return timestamps


def main():
    if not os.path.exists("graphs"):
        os.mkdir("graphs")
    if not os.path.exists(os.path.join("graphs", "celestial-paths")):
        os.mkdir(os.path.join("graphs", "celestial-paths"))

    PLANETS = ['Mercury', 'Venus', 'Mars', "Jupiter", 'Saturn', 'Uranus', "Neptune"]
    PLANET_PRETTY_PERIODS = {
        "Mercury": 355,
        "Venus": 584,
        "Mars": 1000,
        "Jupiter": 4332.589,
        "Saturn": 10759.22,
        "Uranus": 30685.4,
        "Neptune": 60189
    }

    for planet in PLANETS:
        print(f"{planet}...")
        timestamps = get_timestamps(PLANET_PRETTY_PERIODS[planet], 500)
        path = [[], []]
        for timestamp in timestamps:
            ra, dec, r = get_pos(planet, timestamp)
            path[0].append(ra)
            path[1].append(dec)

        plt.figure(figsize=(10, 6))
        plt.title(f"Path of {planet}")
        plt.xlabel("Right Ascension")
        plt.xticks(range(0, 361, 30))
        plt.ylabel("Declination")
        plt.yticks(range(-180, 181, 10))
        plt.scatter(path[0], path[1], label=planet, s=10)
        plt.savefig(os.path.join("graphs", "celestial-paths", f"{planet}.png"))
        plt.close()


if __name__ == "__main__":
    main()