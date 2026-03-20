import numpy as np
import matplotlib.pyplot as plt
import random
import math
from datetime import datetime


class RandGeoModel:
    IDX_ECCENTRIC_ANGLE = 0
    IDX_EPICYCLE_ANGLE = 1
    IDX_PLANET_ANGLE = 2
    IDX_ECCENTRICITY = 3
    IDX_RADII = 4
    IDX_PE_AV = 5
    IDX_ED_AV = 6


    def __init__(self, src: "RGM_Initializer"):
        self.stage = src.stage
        self.start_time = src.start_time
        self.start_long = src.start_long
        self.avg_av = src.avg_av
        self.synodic_av = src.synodic_av
        self.deferent_center = src.deferent_center
        self.properties = src.properties
            

    def predict_pos(self, curr_time: datetime):
        # get current epicycle position
        td = curr_time - self.start_time
        days = td.days + td.seconds/86400
        if type(self.properties[self.IDX_ED_AV]) is tuple:
            print(self.properties)
        curr_ec_angle = self.properties[self.IDX_EPICYCLE_ANGLE] + days*self.properties[self.IDX_ED_AV]
        ec_pos = (
            self.deferent_center[0] + math.cos(curr_ec_angle),
            self.deferent_center[1] + math.sin(curr_ec_angle)
        )

        # get current planet position
        curr_pl_angle = self.properties[self.IDX_PLANET_ANGLE] + days*self.properties[self.IDX_PE_AV]
        pl_pos = (
            ec_pos[0] + self.properties[self.IDX_RADII]*math.cos(curr_pl_angle),
            ec_pos[1] + self.properties[self.IDX_RADII]*math.sin(curr_pl_angle)
        )

        # atan2 to get longitude
        return ec_pos, pl_pos, math.degrees(math.atan2(pl_pos[1], pl_pos[0]))%360


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


    def graph_model(self):
        # compute relevant positions
        epicycle_center, planet_pos, long = self.predict_pos(self.start_time)

        # draw Earth, eccentric, and deferent
        plt.figure(figsize=(6, 6))
        plt.scatter([0], [0], c='b')
        plt.scatter([self.deferent_center[0]], [self.deferent_center[1]], c='r')
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_patch(plt.Circle(self.deferent_center, 1, color='r', fill=False))

        # draw planet and epicycle
        plt.scatter(
            [planet_pos[0]],
            [planet_pos[1]], c='g'
        )
        ax.add_patch(plt.Circle(epicycle_center, self.properties[self.IDX_RADII], color='g', fill=False))

        # optionally draw initial longitude line
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()
        plt.plot(x_limits, [math.tan(math.radians(self.start_long)) * val for val in x_limits])

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



class RGM_Initializer:
    '''
    stage 1: first-gen deferent models
    - manage: eccentricity, eccentric angle, ed_av, radii=0, epicycle angle
    - args: start_time, start_long, avg_av

    stage 2: first-gen epicycle models
    - manage: radii, planet angle, pe_av
    - args: deferent_model, synodic_av

    stage 3: first-gen finetuning models
    - args: epicycle_model
    '''

    IDX_ECCENTRIC_ANGLE = 0
    IDX_EPICYCLE_ANGLE = 1
    IDX_PLANET_ANGLE = 2
    IDX_ECCENTRICITY = 3
    IDX_RADII = 4
    IDX_PE_AV = 5
    IDX_ED_AV = 6

    MAX_ECCENTRICITY = .9
    RADII_BUFFER = .1  # minimum distance between Earth and planet
    MUTATION_RATES = [
        math.radians(5), math.radians(5), math.radians(5),
        .1, .1, .05, .05
    ]
    CROSSOVER_RATE = .2


    def __init__(self):
        self.stage = None
        self.start_time = None
        self.start_long = None
        self.properties = None

        self.__start_eccentric_angle = None
        self.__start_epicycle_angle = None
        self.__start_planet_angle = None
        self.__start_eccentricity = None
        self.__start_radii = None
        self.__start_ed_av = None
        self.__start_pe_av = None

        self.avg_av = None
        self.synodic_av = None
        self.deferent_center = None


    def stage1(self, start_time: datetime, start_long: float, avg_av: float):
        self.stage = 1
        self.start_time = start_time
        self.start_long = start_long
        self.avg_av = avg_av

        self.__start_eccentric_angle = random.random() * 2 * math.pi
        self.__start_eccentricity = .1
        self.deferent_center = (
            self.__start_eccentricity * math.cos(self.__start_eccentric_angle),
            self.__start_eccentricity * math.sin(self.__start_eccentric_angle)
        )
        self.__start_radii = 0
        self.__start_ed_av = .95 * avg_av + .1 * random.random() * avg_av
        self.__start_epicycle_angle = self.guaranteed_epicycle()

        self.properties = [
            self.__start_eccentric_angle,
            self.__start_epicycle_angle,
            0, # self.__start_planet_angle,
            self.__start_eccentricity,
            0, # self.__start_radii,
            0, # self.__start_pe_av,
            self.__start_ed_av,
        ]


    def stage2(self, deferent_model: RandGeoModel, synodic: float):
        self.stage = 2
        # import previously defined
        self.start_time = deferent_model.start_time
        self.start_long = deferent_model.start_long
        self.avg_av = deferent_model.avg_av
        self.synodic_av = 2*math.pi/synodic + self.avg_av

        self.__start_eccentric_angle = deferent_model.properties[self.IDX_ECCENTRIC_ANGLE]
        self.__start_eccentricity = deferent_model.properties[self.IDX_ECCENTRICITY]
        self.deferent_center = deferent_model.deferent_center
        self.__start_ed_av = deferent_model.properties[self.IDX_ED_AV]
        self.__start_epicycle_angle = deferent_model.properties[self.IDX_EPICYCLE_ANGLE]

        # generate random values for new properties
        self.__start_radii = (1 - self.__start_eccentricity - self.RADII_BUFFER) * random.random()
        self.__start_pe_av = .9 * self.synodic_av + .2 * random.random() * self.synodic_av
        self.__start_planet_angle = random.choice([ # assumes pred[0] of deferent_model is accurate
            self.__start_epicycle_angle,
            (self.__start_epicycle_angle+math.pi)%(2*math.pi)
        ])

        self.properties = [
            self.__start_eccentric_angle,
            self.__start_epicycle_angle,
            self.__start_planet_angle,
            self.__start_eccentricity,
            self.__start_radii,
            self.__start_pe_av,
            self.__start_ed_av,
        ]


    def stage3(self, epicycle_model: RandGeoModel):
        self.stage = 3
        self.start_time = epicycle_model.start_time
        self.start_long = epicycle_model.start_long
        self.avg_av = epicycle_model.avg_av
        self.synodic_av = epicycle_model.synodic_av
        self.deferent_center = epicycle_model.deferent_center
        self.properties = epicycle_model.properties


    def child(self, first: RandGeoModel, second: RandGeoModel, mutstr: float):
        self.stage = first.stage
        self.start_time = first.start_time
        self.start_long = first.start_long
        self.avg_av = first.avg_av
        self.synodic_av = first.synodic_av
        self.deferent_center = first.deferent_center
        self.properties = self.crossover(first, second)
        self.mutate(mutstr)


    def mutate(self, strength: float):
        '''
        Makes own genome slightly different from before
        '''
        if self.stage == 1 or self.stage == 3:
            for i in [self.IDX_ECCENTRICITY, self.IDX_ED_AV]:
                noise_range = self.properties[i] * self.MUTATION_RATES[i] * strength * 2
                self.properties[i] += random.random() * noise_range - noise_range / 2
            for i in [self.IDX_ECCENTRIC_ANGLE, self.IDX_EPICYCLE_ANGLE]:
                noise_range = self.MUTATION_RATES[i] * strength * 2
                self.properties[i] += random.random() * noise_range - noise_range / 2
                self.properties[i] %= 2 * math.pi
            self.properties[self.IDX_ECCENTRICITY] = np.clip(
                self.properties[self.IDX_ECCENTRICITY],
                0, self.MAX_ECCENTRICITY
            )
            self.properties[self.IDX_ED_AV] = np.clip(
                self.properties[self.IDX_ED_AV],
                .9 * self.avg_av, 1.1 * self.avg_av
            )

        if self.stage == 2 or self.stage == 3:
            # radii, planet angle, pe_av
            for i in [self.IDX_RADII, self.IDX_PE_AV]:
                noise_range = self.properties[i] * self.MUTATION_RATES[i] * strength * 2
                self.properties[i] += random.random() * noise_range - noise_range / 2
            for i in [self.IDX_PLANET_ANGLE]:
                noise_range = self.MUTATION_RATES[i] * strength * 2
                self.properties[i] += random.random() * noise_range - noise_range / 2
                self.properties[i] %= 2 * math.pi
            self.properties[self.IDX_RADII] = np.clip(
                self.properties[self.IDX_RADII],
                0, 1 - self.properties[self.IDX_ECCENTRICITY] - self.RADII_BUFFER)

            self.properties[self.IDX_PE_AV] = np.clip(
                self.properties[self.IDX_PE_AV],
                .9*self.synodic_av, 1.1*self.synodic_av
            )


    def crossover(self, first: RandGeoModel, second: RandGeoModel):
        '''
        Returns new genome from two models
        '''
        prob_nothing = (1 - self.CROSSOVER_RATE)**len(first.properties)
        if not random.random() < prob_nothing:
            start_swap = random.randint(1, len(first.properties)-1)
            return first.properties[:start_swap] + second.properties[start_swap:]
        return first.properties.copy()
    

    def min_long_diff(self, first: float, second: float):
        # assumes longitudes are in radians
        long_top = max(first, second)
        long_bottom = min(first, second)
        start_diff_frac = min(long_top-long_bottom, (2*math.pi)-long_top+long_bottom) / math.pi
        return start_diff_frac


    def guaranteed_epicycle(self):
        h = self.__start_eccentricity * math.cos(self.__start_eccentric_angle)
        k = self.__start_eccentricity * math.sin(self.__start_eccentric_angle)
        m = math.tan(math.radians(self.start_long))
        a = m ** 2 + 1
        b = -2 * (h + k * m)
        c = h * h + k * k - 1

        x0 = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
        y0 = m * x0
        long0 = math.atan2(y0, x0)%(2*math.pi)
        if self.min_long_diff(math.radians(self.start_long), long0) < .95:
            return math.atan2(y0 - k, x0 - h)%(2*math.pi)
        else:
            x1 = (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
            y1 = m * x1
            return math.atan2(y1 - k, x1 - h)%(2*math.pi)
