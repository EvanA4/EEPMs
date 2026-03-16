import numpy as np
import matplotlib.pyplot as plt
import random
import math
from datetime import datetime


class GeoModel:
    '''
    List of properties by index:
    - angle of eccentric from vernal equinox
    - eccentricity (distance from Earth to eccentric / radius of deferent)
    - radius of epicycle / radius of deferent
    - angular velocity of planet around epicycle / angular velocity of epicycle around deferent
    - starting angle of epicycle on deferent
    - starting angle of planet on epicycle
    '''


    def __init__(self, start_date: datetime, start_long: float):
        self.start_date = start_date
        self.start_long = start_long

        self.eccentric_angle = random.random() * 2 * math.pi
        self.eccentricity = random.random() / 2
        self.radii = random.random() / 4
        self.angular_velocity = random.random()

        # confined such that planet can be at starting longitude
        bounds, offset = self.epicycle_bounds()
        diff = bounds[1] - bounds[0]
        self.epicycle_angle = bounds[0] + random.random()*diff + offset

        # determine planet angle
        h = self.eccentricity * math.cos(self.eccentric_angle) + math.cos(self.epicycle_angle)
        k = self.eccentricity * math.sin(self.eccentric_angle) + math.sin(self.epicycle_angle)
        m = math.tan(self.start_long)
        a = m*m + 1
        b = -2 * (h + k*m)
        c = h*h + k*k - self.radii**2
        sign = -1 if random.random() < .5 else 1
        x = (-b + sign * math.sqrt(b*b - 4*a*c)) / (2*a)
        y = m*x
        self.planet_angle = math.atan2(y-k, x-h)


    def is_valid_epicycle(self, epicycle_angle: float):
        ex = self.eccentricity * math.cos(self.eccentric_angle) + math.cos(epicycle_angle)
        ey = self.eccentricity * math.sin(self.eccentric_angle) + math.sin(epicycle_angle)
        if (ex + ey * math.tan(self.start_long) < 0): return False
        a = -math.tan(self.start_long)
        sq_distance = (a * ex + ey)**2 / (a*a + 1)
        return sq_distance < self.radii**2

    
    def guaranteed_epicycle(self):
        h = self.eccentricity * math.cos(self.eccentric_angle)
        k = self.eccentricity * math.sin(self.eccentric_angle)
        m = math.tan(self.start_long)
        a = m**2 + 1
        b = -2 * (h + k*m)
        c = h*h + k*k - 1
        
        x0 = (-b + math.sqrt(b*b - 4*a*c)) / (2*a)
        y0 = m*x0
        if (x0 + math.tan(self.start_long)*y0 > 0):
            return math.atan2(y0-k, x0-h)
        else:
            x1 = (-b - math.sqrt(b*b - 4*a*c)) / (2*a)
            y1 = m*x1
            return math.atan2(y1-k, x1-h)

    
    def epicycle_bounds(self):
        NUM_ITERATIONS = 10
        offset = self.guaranteed_epicycle() - math.pi # prevents angle-wrapping shenanigans
        bounds = [None, None]
        
        # find CW point first
        ccwptr = math.pi
        cwptr = math.pi/2
        for i in range(NUM_ITERATIONS):
            mid = (ccwptr + cwptr) / 2
            if (self.is_valid_epicycle(mid + offset)):
                ccwptr = mid
            else:
                cwptr = mid
        bounds[0] = ccwptr

        # then find CCW point
        ccwptr = 3*math.pi/2
        cwptr = math.pi
        for i in range(NUM_ITERATIONS):
            mid = (ccwptr + cwptr) / 2
            if (self.is_valid_epicycle(mid + offset)):
                cwptr = mid
            else:
                ccwptr = mid
        bounds[1] = cwptr
        return bounds, offset


    def graph_params(self):
        # display min and max
        plt.figure(figsize=(6, 6))
        plt.scatter([0], [0], c='b')
        plt.scatter([self.eccentricity * math.cos(self.eccentric_angle)], [self.eccentricity * math.sin(self.eccentric_angle)], c='r')
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_patch(
            plt.Circle((
                self.eccentricity * math.cos(self.eccentric_angle),
                self.eccentricity * math.sin(self.eccentric_angle)
            ), 1, color='r', fill=False)
        )
        plt.scatter(
            [self.eccentricity * math.cos(self.eccentric_angle) + math.cos(self.epicycle_angle) + self.radii*math.cos(self.planet_angle)],
            [self.eccentricity * math.sin(self.eccentric_angle) + math.sin(self.epicycle_angle) + self.radii*math.sin(self.planet_angle)], c='g'
        )
        ax.add_patch(
            plt.Circle((
                self.eccentricity * math.cos(self.eccentric_angle) + math.cos(self.epicycle_angle),
                self.eccentricity * math.sin(self.eccentric_angle) + math.sin(self.epicycle_angle)
            ), self.radii, color='g', fill=False)
        )

        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()

        plt.plot(x_limits, [math.tan(self.start_long) * val for val in x_limits])

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        plot_radius = 0.5*max([x_range, y_range])
        ax.set_xlim([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim([y_middle - plot_radius, y_middle + plot_radius])

        plt.show()
        plt.close()