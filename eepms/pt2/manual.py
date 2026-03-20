from geomodel import RandGeoModel
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sys import argv, stderr
import math
import numpy as np


PLANETS = ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"]
COMPUTED = {
    "mercury": [
        3.606343102378771,
        4.884345283434743,
        3.037667411254878,
        0.0994981288861486,
        0.3852665339416921,
        0.07151937481630387,
        0.017361394566852165,
    ],
    "venus": [
        0.7362462892451473,
        5.2181212953632174,
        0.248052768785641,
        0.08545405869593471,
        0.7176540373023494,
        0.02793860062568779,
        0.017135171922363156,
    ],
    "mars": [
        2.251063731719305,
        1.00313819000623,
        4.725928844472276,
        0.10310843081435339,
        0.6630390873658489,
        0.01723967495680901,
        0.009174051062618753,
    ],
    "jupiter": [
        3.227599357560533,
        2.0303910811090224,
        4.939135788733546,
        0.08915066344298914,
        0.19041805326011224,
        0.017190275626483474,
        0.0014388887891256055,
    ],
    "saturn": [
        3.700277436926513,
        2.3997389170190027,
        4.9427367625092105,
        0.11512766314457303,
        0.11143805178619554,
        0.017198480423003676,
        0.0006106160239563701,
    ],
    "uranus": [
        2.356092104807293,
        3.2508367227533395,
        5.123039487876838,
        0.08254065821958605,
        0.055332578238235806,
        0.01719671042684957,
        0.00022431004990504226,
    ],
    "neptune": [
        3.7662003165440705,
        4.051225300855737,
        4.953585665957901,
        0.11365081452112073,
        0.035624850830340664,
        0.017192999810501666,
        0.00011075059190786893,
    ]
}


def predict(props: list[float], start_time: datetime, dt: datetime):
    # get current epicycle position
    td = dt - start_time
    days = td.days + td.seconds / 86400
    if type(props[RandGeoModel.IDX_ED_AV]) is tuple:
        print(props)
    curr_ec_angle = props[RandGeoModel.IDX_EPICYCLE_ANGLE] + days * props[RandGeoModel.IDX_ED_AV]
    deferent_center = (
        props[RandGeoModel.IDX_ECCENTRICITY] * math.cos(props[RandGeoModel.IDX_ECCENTRIC_ANGLE]),
        props[RandGeoModel.IDX_ECCENTRICITY] * math.sin(props[RandGeoModel.IDX_ECCENTRIC_ANGLE])
    )
    ec_pos = (
        deferent_center[0] + math.cos(curr_ec_angle),
        deferent_center[1] + math.sin(curr_ec_angle)
    )

    # get current planet position
    curr_pl_angle = props[RandGeoModel.IDX_PLANET_ANGLE] + days * props[RandGeoModel.IDX_PE_AV]
    pl_pos = (
        ec_pos[0] + props[RandGeoModel.IDX_RADII] * math.cos(curr_pl_angle),
        ec_pos[1] + props[RandGeoModel.IDX_RADII] * math.sin(curr_pl_angle)
    )

    # atan2 to get longitude
    return ec_pos, pl_pos, math.degrees(math.atan2(pl_pos[1], pl_pos[0])) % 360


def main():
    if len(argv) != 2 or argv[1] not in PLANETS:
        print(f"usage: {argv[0]} planet", file=stderr)
        return 1

    # read data
    df = pd.read_csv(os.path.join("csvs", "expected", f"{argv[1]}.csv"))
    datetimes = [datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") for timestamp in df["Timestamp"].to_list()]
    longitudes = df["Longitude"].to_list()
    props = COMPUTED[argv[1]]
    preds = [predict(props, datetimes[0], dt)[2] for dt in datetimes]

    # compute relevant positions
    epicycle_center, planet_pos, long = predict(props, datetimes[0], datetimes[0])
    deferent_center = (
        props[RandGeoModel.IDX_ECCENTRICITY] * math.cos(props[RandGeoModel.IDX_ECCENTRIC_ANGLE]),
        props[RandGeoModel.IDX_ECCENTRICITY] * math.sin(props[RandGeoModel.IDX_ECCENTRIC_ANGLE])
    )

    # draw Earth, eccentric, and deferent
    plt.figure(figsize=(6, 6))
    plt.scatter([0], [0], c='b')
    plt.scatter([deferent_center[0]], [deferent_center[1]], c='r')
    fig = plt.gcf()
    ax = fig.gca()
    ax.add_patch(plt.Circle(deferent_center, 1, color='r', fill=False))

    # draw planet and epicycle
    plt.scatter(
        [planet_pos[0]],
        [planet_pos[1]], c='g'
    )
    ax.add_patch(plt.Circle(epicycle_center, props[RandGeoModel.IDX_RADII], color='g', fill=False))

    # optionally draw initial longitude line
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    plt.plot(x_limits, [math.tan(math.radians(preds[0])) * val for val in x_limits])

    # set axes equal
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    plot_radius = 0.5*max([x_range, y_range])
    ax.set_xlim([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim([y_middle - plot_radius, y_middle + plot_radius])

    plt.show()
    plt.close()

    # Expected Planetary Path
    plt.figure(figsize=(10, 6))
    plt.title(f"Longitude vs. Time for {argv[1][0].upper() + argv[1][1:]}")
    plt.xlabel("Date")
    plt.ylabel("Longitude")
    plt.yticks(range(0,361,30))
    plt.scatter(datetimes, longitudes, s=10, c='g', label="Expected", alpha=.5)
    plt.scatter(datetimes, preds, s=10, c='r', label="Actual", alpha=.5)
    plt.legend()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()