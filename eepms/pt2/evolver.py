import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import math
from datetime import datetime

from helpers import get_retro_times, get_retro_stats, get_cumu_longs, min_long_diff, first_retro
from geomodel import RandGeoModel, RGM_Initializer


class RGM_Evolver:
    GEN_SIZE = 1000
    TOURN_SIZE = 10
    NUM_GENS = [0, 10, 30, 5]
    PERFECT = 10
    PENALTY = 1
    SMALL_EPICYCLE = .1
    STRENGTHS = [0, 1, 1, .2]


    def __init__(self):
        # read data
        self.gen: list[RandGeoModel] = []
        df = pd.read_csv(os.path.join("csvs", "expected", "mars.csv"))
        self.datetimes = [datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") for timestamp in df["Timestamp"].to_list()]
        self.longitudes = df["Longitude"].to_list()
        self.cumu_longs = get_cumu_longs(self.longitudes)
        self.stage = 1

        # approximate average angular velocity of planet
        self.longitude_range = math.radians(self.cumu_longs[-1] - self.cumu_longs[0])
        td = self.datetimes[-1] - self.datetimes[0]
        days = td.days + td.seconds/86400
        self.avg_av = self.longitude_range / days

        # compute retrograde descriptors
        self.retro_times = get_retro_times(self.cumu_longs)
        self.retrogap, self.retrolen, self.retroheight = get_retro_stats(
            self.retro_times, self.datetimes, self.cumu_longs
        )
        self.retro_offset = first_retro(self.retro_times, self.datetimes)

        # # find average drop in longitude with each step, only looking at retrograde (in degrees)
        # self.retrostep, self.prostep = get_steps(self.cumu_longs)

        print(
            f"Longitude range:          {self.longitude_range/(2*math.pi)}\n"
            f"Average angular velocity: {self.avg_av}\n"
            f"Retrograde stats:\n"
            f"    retrogap:             {self.retrogap}\n"
            f"    retrolen:             {self.retrolen}\n"
            f"    retroheight:          {self.retroheight}\n"
            # f"Avg Retrostep:            {self.retrostep}\n"
            # f"Avg Prostep:              {self.prostep}\n"
        )


    def model_eval(self, model: RandGeoModel, verbose=False):
        # mean squared error
        preds = [model.predict_pos(dt)[2] for dt in self.datetimes]
        if self.stage == 1:
            errs = [min_long_diff(preds[i], self.longitudes[i], False) for i in range(len(self.datetimes))]
            sq_errs = [err**2 for err in errs]
            msqe_penalty = float(np.mean(sq_errs) * self.PENALTY * 5)
            if verbose:
                print(f"msqe_penalty: {msqe_penalty}\n")
            return self.PERFECT - msqe_penalty

        # averaged retrograde statistics
        cumu_preds = get_cumu_longs(preds)
        pred_times = get_retro_times(cumu_preds)
        predgap, predlen, predheight = get_retro_stats(pred_times, self.datetimes, cumu_preds)
        retrogap_penalty = math.fabs(predgap-self.retrogap) / self.retrogap * self.PENALTY
        retrolen_penalty = math.fabs(predlen-self.retrolen) / self.retrolen * self.PENALTY
        retroheight_penalty = math.fabs(predheight-self.retroheight) / self.retroheight * self.PENALTY

        # time until first retrograde
        pred_offset = first_retro(pred_times, self.datetimes)
        offset_penalty = math.fabs(pred_offset-self.retro_offset) / self.retro_offset * self.PENALTY

        if verbose:
            print(
                f"retrogap_penalty:    {retrogap_penalty}\n"
                f"retrolen_penalty:    {retrolen_penalty}\n"
                f"retroheight_penalty: {retroheight_penalty}\n"
                f"offset_penalty:      {offset_penalty}\n"
            )
        return self.PERFECT - retrolen_penalty - retroheight_penalty - retrogap_penalty - offset_penalty


    def reproduce(self, tourn: list[tuple[RandGeoModel, float]]):
        strength = self.STRENGTHS[self.stage]
        output = []
        for _ in range(self.GEN_SIZE):
            first = random.randint(0, len(tourn)-1)
            second = random.randint(0, len(tourn)-1)
            ini = RGM_Initializer()
            ini.child(tourn[first][0], tourn[second][0], strength)
            output.append(RandGeoModel(ini))
        return output


    def simulate(self, stage: int, best_model=None):
        # populate first generation
        self.stage = stage
        self.gen = []
        for _ in range(self.GEN_SIZE):
            ini = RGM_Initializer()
            if stage == 1:
                ini.stage1(self.datetimes[0], self.longitudes[0], self.avg_av)
            elif stage == 2:
                ini.stage2(best_model, self.retrogap)
            else:
                ini.stage3(best_model)
            self.gen.append(RandGeoModel(ini))

        gen_results: list[list[tuple[RandGeoModel, float]]] = []
        
        print("Gen | Fitness", flush=True)
        for i in range(self.NUM_GENS[stage]):
            print(f"{i:<3d}", end="", flush=True)
            gen_data = [
                (self.gen[i], self.model_eval(self.gen[i]))
                for i in range(self.GEN_SIZE)
            ]
            gen_data.sort(key=lambda x: x[1], reverse=True)

            tourn = gen_data[:self.TOURN_SIZE]
            gen_results.append(tourn)

            if i != self.NUM_GENS[stage]-1:
                self.gen = self.reproduce(tourn)

            print(f" | {tourn[0][1]:>7.4f}", flush=True)
        print()

        # Loss vs. Generation
        plt.figure(figsize=(10, 6))
        plt.title("Fitness vs. Generation")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        fitnesses = [result[0][1] for result in gen_results]
        plt.plot(range(self.NUM_GENS[stage]), fitnesses)
        plt.show()
        plt.close()

        # print and graph best model
        best_model = gen_results[-1][0][0]
        best_model.print_props()
        best_model.graph_model()
        self.model_eval(best_model, True)

        # Expected Planetary Path
        plt.figure(figsize=(10, 6))
        plt.title("Planetary Paths")
        plt.xlabel("Date")
        plt.ylabel("Longitude")
        plt.yticks(range(0, 361, 30))
        plt.scatter(self.datetimes, self.longitudes, s=10, c='g', label="Expected", alpha=.5)
        plt.scatter(
            self.datetimes, [best_model.predict_pos(dt)[2] for dt in self.datetimes],
            s=10, c='r', label="Actual", alpha=.5
        )
        plt.legend()
        plt.show()
        plt.close()

        return gen_results[-1][0]