from geomodel import RandGeoModel
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# PROPS = [ # mercury
#     0.2101316943653704,   # eccentric_angle
#     4.804683010644778,    # epicycle_angle
#     3.05623590392605,     # planet_angle
#     0.10466683089269988,  # eccentricity
#     0.36235575623662544,  # radii
#     0.06573663483894383,  # pe_av
#     0.016596681851442233, # ed_av
# ]

PROPS = [ # jupiter
    3.6068264511673194,    # eccentric_angle
    1.9884574702567521,    # epicycle_angle
    4.867850650528798,     # planet_angle
    0.07738865135203035,   # eccentricity
    0.19123429167371406,   # 0.19027502279005692   # radii
    0.017249613205693678,  # pe_av
    0.0014625219179870727, # ed_av
]


def main():

    # read data
    df = pd.read_csv(os.path.join("csvs", "expected", "jupiter.csv"))
    datetimes = [datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") for timestamp in df["Timestamp"].to_list()][::4]
    longitudes = df["Longitude"][::4].to_list()

    # print and graph best model
    best_model = RandGeoModel(
        datetimes[0], longitudes[0], 0, 0, PROPS
    )

    # Expected Planetary Path
    plt.figure(figsize=(10, 6))
    plt.title("Planetary Paths")
    plt.xlabel("Date")
    plt.ylabel("Longitude")
    plt.yticks(range(0,361,30))
    plt.scatter(datetimes, longitudes, s=10, c='g', label="Expected", alpha=.5)
    plt.scatter(datetimes, [best_model.predict_pos(dt)[2] for dt in datetimes], s=10, c='r', label="Actual", alpha=.5)
    plt.legend()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()


'''
TODO evolve deferent first (eccentricity, eccentric angle, ed_av, radii=0, epicycle angle) with msqe
TODO evolve epicycle second (radii, planet angle, pe_av)
TODO maybe final pass over all parameters with very tiny mutation?
'''