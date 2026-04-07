from astropy.time import Time
from astropy.coordinates import get_body_barycentric
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
    

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


def draw_lat_circle(ax: plt.Axes, fig: plt.Figure, lat: float):
    NUM_POINTS = 100
    angles = np.linspace(0, 2*math.pi, num=NUM_POINTS+1)
    radius = math.cos(math.radians(lat))
    height = math.sin(math.radians(lat))
    xs = [math.cos(angle)*radius for angle in angles]
    ys = [math.sin(angle)*radius for angle in angles]
    zs = [height for angle in angles]
    if lat == 0:
        ax.plot(xs, ys, zs, c="red", label="Equator")
    else:
        ax.plot(xs, ys, zs, c="lightgray")


def draw_long_circle(ax: plt.Axes, fig: plt.Figure, long: float):
    NUM_POINTS = 100
    angles = np.linspace(0, math.pi*2, NUM_POINTS)   # lat from -90° to +90°
    lon_rad = math.radians(long)
    xs = [math.cos(lat) * math.cos(lon_rad) for lat in angles]
    ys = [math.cos(lat) * math.sin(lon_rad) for lat in angles]
    zs = [math.sin(lat) for lat in angles]
    ax.plot(xs, ys, zs, c="lightgray")


def draw_ecliptic(ax: plt.Axes, fig: plt.Figure):
    """
    Draw the ecliptic (unit circle tilted by obliquity_deg) on a 3D axes.
    plot_kwargs passed to ax.plot (e.g., c='orange', lw=1).
    """
    NUM_POINTS = 100
    OBLIQUITY = 23.4392911
    lambdas = np.linspace(0, 2*math.pi, num=NUM_POINTS+1)
    eps = math.radians(OBLIQUITY)
    xs = np.cos(lambdas)
    ys = np.cos(eps) * np.sin(lambdas)
    zs = np.sin(eps) * np.sin(lambdas)
    ax.plot(xs, ys, zs, c="b", label="Ecliptic")


def main():
    PLANETS = ['Mercury', 'Venus', 'Earth', 'Mars', "Jupiter", 'Saturn', 'Uranus', "Neptune"]
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
    plt.title(f"Path of Planets")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect((1, 1, 1))

    ax.scatter(
        0, 0, 0,
        c='g',
        label="Earth",
        s=80
    )
    
    # draw circles
    longs = range(0, 361, 30)
    for long in longs:
        draw_long_circle(ax, fig, long)
    lats = range(-75, 90, 15)
    for lat in lats:
        draw_lat_circle(ax, fig, lat)
    draw_ecliptic(ax, fig)

    set_axes_equal(ax)
    ax.grid(visible=False)
    ax.set_axis_off()
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()