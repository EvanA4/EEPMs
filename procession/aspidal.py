from astropy.time import Time
from astropy.coordinates import get_body_barycentric
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
    

def get_pos(name: str, timestamp: str):
    t = Time(timestamp)
    cartrep = get_body_barycentric(name, t)
    oc = cartrep.get_xyz().to_value()
    return oc


def get_timestamps(period: float, num_timestamps=10):
    timestamps = []
    dt = datetime.datetime(year=1801, month=1, day=1)    
    timestamps.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
    step_days = period / (num_timestamps - 1)
    for i in range(num_timestamps - 1):
        dt += datetime.timedelta(days=step_days)
        timestamps.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
    return timestamps


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def main():
    PLANET = "Venus"
    NUM_ORBITS = 10
    RESOLUTION = 500
    # PLANETS = ['Mercury', 'Venus', 'Earth', 'Mars']    
    PLANET_SIDERIAL_PERIODS = {
        "Mercury": 87.969,
        "Venus": 224.701,
        "Earth": 365.256,
        "Mars": 686.980,
        "Jupiter": 4332.589,
        "Saturn": 10759.22,
        "Uranus": 30685.4,
        "Neptune": 60189
    }

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.title(f"Path of {PLANET}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect((1, 1, 1))

    ax.scatter(
        0, 0, 0,
        label="Sun",
        s=40,
        c="g"
    )

    # x rotation by 23.433
    theta = np.radians(23.433)
    rotMat = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

    path = [[], [], []]
    timestamps = get_timestamps(PLANET_SIDERIAL_PERIODS[PLANET]*NUM_ORBITS, NUM_ORBITS*RESOLUTION)
    for timestamp in timestamps:
        pos = get_pos(PLANET, timestamp)
        rot_pos = np.matmul(pos, rotMat)
        for i in range(3):
            path[i].append(rot_pos[i])
    ax.plot3D(path[0], path[1], path[2], label=PLANET, c="b")

    set_axes_equal(ax)

    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()