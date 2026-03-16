from astropy.time import Time
from astropy.coordinates import get_body_barycentric
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import math


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
    cartrep = get_body_barycentric("earth", t)
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


def equatorial_to_ecliptic(ra, dec):
    # https://en.wikipedia.org/wiki/Astronomical_coordinate_systems#Ecliptic_system
    OBLIQUITY = math.radians(23.439292)
    ra_rad = math.radians(ra)
    dec_rad = math.radians(dec)
    ecl_long = math.atan2(
        math.sin(ra_rad) * math.cos(OBLIQUITY) +
        math.tan(dec_rad) * math.sin(OBLIQUITY),
        math.cos(ra_rad)
    )
    ecl_lat = math.asin(
        math.sin(dec_rad) * math.cos(OBLIQUITY) -
        math.cos(dec_rad) * math.sin(OBLIQUITY) * math.sin(ra_rad)
    )
    ecl_long_deg = math.degrees(ecl_long)
    ecl_lat_deg = math.degrees(ecl_lat)
    ecl_long_deg = ecl_long_deg % 360

    return ecl_long_deg, ecl_lat_deg


def main():
    if not os.path.exists("graphs"):
        os.mkdir("graphs")
    if not os.path.exists("csvs"):
        os.mkdir("csvs")
    if not os.path.exists(os.path.join("csvs", "expected")):
        os.mkdir(os.path.join("csvs", "expected"))

    PLANETS = ["Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
    PLANET_SYNODIC_PERIODS = {
        "Mercury": 116,
        "Venus": 584,
        "Mars": 780,
        "Jupiter": 399,
        "Saturn": 378,
        "Uranus": 370,
        "Neptune": 367
    }
    NUM_CYCLES = 10
    RESOLUTION = 100

    plt.figure(figsize=(10, 6))
    plt.title(f"Expected Positions for Each Planet")
    plt.xlabel("Ecliptic Longitude")
    plt.xticks(range(0, 361, 30))
    plt.xlim((0, 360))
    plt.ylabel("Ecliptic Latitude")
    plt.yticks(range(-90, 90, 10))
    plt.ylim((-90, 90))

    for planet in PLANETS:
        print(f"{planet}...")
        csv_file = open(os.path.join("csvs", "expected", f"{planet.lower()}.csv"), "w")
        csv_file.write("Timestamp,Longitude,Latitude\n")
        
        timestamps = get_timestamps(PLANET_SYNODIC_PERIODS[planet] * NUM_CYCLES, NUM_CYCLES * RESOLUTION)
        path = [[], []]
        for timestamp in timestamps:
            ra, dec, r = get_pos(planet, timestamp) # equatorial coordinates
            long, lat = equatorial_to_ecliptic(ra, dec) # ecliptic coordinates
            csv_file.write(f"{timestamp},{long},{lat}\n")
            path[0].append(long)
            path[1].append(lat)
            
        plt.scatter(path[0], path[1], label=planet, s=10)

    plt.legend()
    plt.savefig(os.path.join("graphs", "gen-expected.png"))
    plt.close()


if __name__ == "__main__":
    main()
