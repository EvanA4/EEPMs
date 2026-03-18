import numpy as np
import matplotlib.pyplot as plt
import random
import math
from datetime import datetime


class RandGeoModel:
    '''
    List of properties by index:
    - angle of eccentric from vernal equinox
    - eccentricity (distance from Earth to eccentric / radius of deferent)
    - radius of epicycle / radius of deferent
    - angular velocity of planet around epicycle
    - angular velocity of epicycle around deferent
    - starting angle of epicycle on deferent
    - starting angle of planet on epicycle
    '''
    IDX_ECCENTRIC_ANGLE = 0
    IDX_EPICYCLE_ANGLE = 1
    IDX_PLANET_ANGLE = 2
    IDX_ECCENTRICITY = 3
    IDX_RADII = 4
    IDX_PE_AV = 5
    IDX_ED_AV = 6
    MAX_ECCENTRICITY = .9
    RADII_BUFFER = .1 # minimum distance between Earth and planet
    MUTATION_RATES = [math.radians(2), math.radians(2), math.radians(2), .1, .1, .1, .1]
    CROSSOVER_RATE = .2


    def __init__(self, start_time: datetime, start_long: float, avg_av: float, synodic_av: float, properties=None):
        self.start_time = start_time
        self.start_long = start_long
        self.avg_av = avg_av
        self.synodic_av = synodic_av

        if properties != None:
            self.properties = properties
            self.__deferent_center = (
                self.properties[self.IDX_ECCENTRICITY] * math.cos(self.properties[self.IDX_ECCENTRIC_ANGLE]),
                self.properties[self.IDX_ECCENTRICITY] * math.sin(self.properties[self.IDX_ECCENTRIC_ANGLE])
            )

        else:
            self.__start_eccentric_angle = random.random() * 2 * math.pi
            self.__start_eccentricity = random.random() * self.MAX_ECCENTRICITY
            self.__deferent_center = (
                self.__start_eccentricity * math.cos(self.__start_eccentric_angle),
                self.__start_eccentricity * math.sin(self.__start_eccentric_angle)
            )
            self.__start_radii = (1 - self.__start_eccentricity - self.RADII_BUFFER) * random.random()
            self.__start_pe_av = .8*synodic_av + .4*random.random()*synodic_av
            self.__start_ed_av = .8*avg_av + .4*random.random()*avg_av

            # confined such that planet can be at starting longitude
            bounds, offset = self.epicycle_bounds()
            diff = bounds[1] - bounds[0]
            self.__start_epicycle_angle = bounds[0] + random.random()*diff + offset

            # determine planet angle
            h = self.__start_eccentricity * math.cos(self.__start_eccentric_angle) + math.cos(self.__start_epicycle_angle)
            k = self.__start_eccentricity * math.sin(self.__start_eccentric_angle) + math.sin(self.__start_epicycle_angle)
            m = math.tan(self.start_long)
            a = m*m + 1
            b = -2 * (h + k*m)
            c = h*h + k*k - self.__start_radii**2
            sign = -1 if random.random() < .5 else 1
            x = (-b + sign * math.sqrt(b*b - 4*a*c)) / (2*a)
            y = m*x
            self.__start_planet_angle = math.atan2(y-k, x-h)
            
            self.properties = [
                self.__start_eccentric_angle,
                self.__start_epicycle_angle,
                self.__start_planet_angle,
                self.__start_eccentricity,
                self.__start_radii,
                self.__start_pe_av,
                self.__start_ed_av,
            ]


    def mutate(self, strength: float):
        '''
        Makes own genome slightly different from before
        '''
        FIRST_NONANGLE_IDX = 3
        for i in range(FIRST_NONANGLE_IDX):
            self.properties[i] = (self.properties[i] + random.random()*2*self.MUTATION_RATES[i]*strength - self.MUTATION_RATES[i]*strength) % (2*math.pi)


        for i in range(FIRST_NONANGLE_IDX, len(self.properties)):
            self.properties[i] +=                                                                            \
                random.random() * self.properties[i] * self.MUTATION_RATES[i-FIRST_NONANGLE_IDX]*strength*2 - \
                self.properties[i] * self.MUTATION_RATES[i-FIRST_NONANGLE_IDX]*strength

        bounds = [
            (0, self.MAX_ECCENTRICITY),
            (0, 1 - self.properties[self.IDX_ECCENTRICITY] - self.RADII_BUFFER),
            (.8*self.synodic_av, 1.2*self.synodic_av),
            (.8*self.avg_av, 1.2*self.avg_av),
        ]
        for i in range(3, len(self.properties)):
            if self.properties[i] < bounds[i-FIRST_NONANGLE_IDX][0]:
                self.properties[i] = bounds[i-FIRST_NONANGLE_IDX][0]
            elif self.properties[i] > bounds[i-FIRST_NONANGLE_IDX][1]:
                self.properties[i] = bounds[i-FIRST_NONANGLE_IDX][1]


    def crossover(self, model: "RandGeoModel"):
        '''
        Returns new genome from two models
        '''
        prob_nothing = (1 - self.CROSSOVER_RATE)**len(self.properties)
        if not random.random() < prob_nothing:
            start_swap = random.randint(1, len(self.properties)-1)
            return self.properties[:start_swap] + model.properties[start_swap:]
        return self.properties.copy()
            


    def predict_pos(self, curr_time: datetime):
        # get current epicycle position
        td = curr_time - self.start_time
        days = td.days + td.seconds/86400
        curr_ec_angle = self.properties[self.IDX_EPICYCLE_ANGLE] + days*self.properties[self.IDX_ED_AV]
        ec_pos = (
            self.__deferent_center[0] + math.cos(curr_ec_angle),
            self.__deferent_center[1] + math.sin(curr_ec_angle)
        )

        # get current planet position
        curr_pl_angle = self.properties[self.IDX_PLANET_ANGLE] + days*self.properties[self.IDX_PE_AV]
        pl_pos = (
            ec_pos[0] + self.properties[self.IDX_RADII]*math.cos(curr_pl_angle),
            ec_pos[1] + self.properties[self.IDX_RADII]*math.sin(curr_pl_angle)
        )

        # atan2 to get longitude
        return ec_pos, pl_pos, math.degrees(math.atan2(pl_pos[1], pl_pos[0]))%360


    def is_valid_epicycle(self, epicycle_angle: float):
        ex = self.__start_eccentricity * math.cos(self.__start_eccentric_angle) + math.cos(epicycle_angle)
        ey = self.__start_eccentricity * math.sin(self.__start_eccentric_angle) + math.sin(epicycle_angle)
        if (ex + ey * math.tan(self.start_long) < 0): return False
        a = -math.tan(self.start_long)
        sq_distance = (a * ex + ey)**2 / (a*a + 1)
        return sq_distance < self.__start_radii**2

    
    def guaranteed_epicycle(self):
        h = self.__start_eccentricity * math.cos(self.__start_eccentric_angle)
        k = self.__start_eccentricity * math.sin(self.__start_eccentric_angle)
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


    def print_props(self):
        print(
            f"ECCENTRIC_ANGLE: {self.properties[self.IDX_ECCENTRIC_ANGLE]}\n"
            f"EPICYCLE_ANGLE:  {self.properties[self.IDX_EPICYCLE_ANGLE]}\n"
            f"PLANET_ANGLE:    {self.properties[self.IDX_PLANET_ANGLE]}\n"
            f"ECCENTRICITY:    {self.properties[self.IDX_ECCENTRICITY]}\n"
            f"RADII:           {self.properties[self.IDX_RADII]}\n"
            f"PE_AV:           {self.properties[self.IDX_PE_AV]}\n"
            f"ED_AV:           {self.properties[self.IDX_ED_AV]}\n"
        )


    def graph_model(self, curr_time=None):
        # compute relevant positions
        epicycle_center, planet_pos, long = self.predict_pos(curr_time if curr_time else self.start_time)

        # draw Earth, eccentric, and deferent
        plt.figure(figsize=(6, 6))
        plt.scatter([0], [0], c='b')
        plt.scatter([self.__deferent_center[0]], [self.__deferent_center[1]], c='r')
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_patch(plt.Circle(self.__deferent_center, 1, color='r', fill=False))

        # draw planet and epicycle
        plt.scatter(
            [planet_pos[0]],
            [planet_pos[1]], c='g'
        )
        ax.add_patch(plt.Circle(epicycle_center, self.properties[self.IDX_RADII], color='g', fill=False))

        # optionally draw initial longitude line
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()
        if curr_time == None:
            plt.plot(x_limits, [math.tan(self.start_long) * val for val in x_limits])

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