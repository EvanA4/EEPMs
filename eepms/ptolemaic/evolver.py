import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import math
from datetime import datetime, timedelta
from geomodel import RandGeoModel


class RGM_Evolver:
    GEN_SIZE = 1000
    TOURN_SIZE = 10
    NUM_GENS = 10
    PERFECT = 10
    PENALTY = 1
    SMALL_EPICYCLE = .1


    def __init__(self):
        # read data
        df = pd.read_csv(os.path.join("csvs", "expected", "mercury.csv"))
        self.datetimes = [datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") for timestamp in df["Timestamp"].to_list()][::4]
        self.longitudes = df["Longitude"][::4].to_list()

        # approximate average angular velocity of planet
        self.longitude_range = math.radians(self.get_long_range(self.longitudes))
        TIME_DELTA = self.datetimes[-1] - self.datetimes[0]
        DAYS_RANGE = TIME_DELTA.days + TIME_DELTA.seconds/86400
        self.avg_av = self.longitude_range / DAYS_RANGE
        self.max_msqe = 180**2 * len(self.datetimes)

        # populate first generation
        self.gen = [
            RandGeoModel(self.datetimes[0], self.longitudes[0], self.avg_av)
            for _ in range(self.GEN_SIZE)
        ]


    def get_long_range(self, longs: list[float], is_radians=False):
        # assumes longitudes are in degrees
        HALF_JUMP = math.pi if is_radians else 180
        FULL_JUMP = 2*HALF_JUMP
        offset = 0
        for i in range(1, len(longs)):
            wrapped_down = longs[i]-longs[i-1] < -HALF_JUMP
            wrapped_up = longs[i]-longs[i-1] > HALF_JUMP
            if wrapped_down: offset += FULL_JUMP
            elif wrapped_up: offset -= FULL_JUMP
        return longs[-1] - longs[0] + offset


    def min_long_diff(self, first, second):
        # assumes longitudes are in degrees
        long_top = max(first, second)
        long_bottom = min(first, second)
        start_diff_frac = min(long_top-long_bottom, 360-long_top+long_bottom) / 180
        return start_diff_frac


    def model_eval(self, model: RandGeoModel, verbose=False):
        # mean squared error
        preds = [model.predict_pos(dt)[2] for dt in self.datetimes]
        sq_losses = [(preds[i]-self.longitudes[i])**2 for i in range(len(self.datetimes))]
        msqe_penalty = np.mean(sq_losses) / self.max_msqe * self.PENALTY

        # starting longitude should be the same
        start_penalty = self.min_long_diff(preds[0], self.longitudes[0]) / 2 * self.PENALTY

        # planet's longitude should change the same amount
        preds_range = math.radians(self.get_long_range(preds))
        long_penalty = math.fabs(preds_range-self.longitude_range) / self.longitude_range * self.PENALTY

        # epicycle-around-deferent speed should be around average angular speed of planet
        stagnant_penalty = math.fabs(model.properties[RandGeoModel.IDX_ED_AV]-self.avg_av)/self.avg_av * self.PENALTY

        # unreasonably small epicycle
        radii_penalty = self.PENALTY if model.properties[RandGeoModel.IDX_RADII] < self.SMALL_EPICYCLE else 0

        if verbose:
            print(
                f"msqe_penalty:     {msqe_penalty}\n"
                f"start_penalty:    {start_penalty}\n"
                f"stagnant_penalty: {stagnant_penalty}\n"
                f"radii_penalty:    {radii_penalty}\n"
                f"long_penalty:     {long_penalty}\n"
            )

        return self.PERFECT - (msqe_penalty + start_penalty + long_penalty + stagnant_penalty + radii_penalty)


    def reproduce(self, tourn, GEN_SIZE):
        output = []
        for _ in range(GEN_SIZE):
            first = random.randint(0, len(tourn)-1)
            second = random.randint(0, len(tourn)-1)
            model = RandGeoModel(
                tourn[first][0].start_time,
                tourn[first][0].start_long,
                self.avg_av,
                tourn[first][0].crossover(tourn[second][0])
            )
            model.mutate()
            output.append(model)
        return output


    def simulate(self):
        gen_results: list[tuple[list[RandGeoModel], list[float]]] = []
        tourn: list[tuple[RandGeoModel, float]] = []
        
        print("Gen | Fitness", flush=True)
        for i in range(self.NUM_GENS):
            print(f"{i:<3d}", end="", flush=True)
            gen_data = [
                (self.gen[i], self.model_eval(self.gen[i]))
                for i in range(self.GEN_SIZE)
            ]
            gen_data.sort(key=lambda x: x[1], reverse=True)

            tourn = gen_data[:self.TOURN_SIZE]
            gen_results.append(tourn)

            if i != self.NUM_GENS-1:
                self.gen = self.reproduce(tourn, self.GEN_SIZE)

            print(f" | {tourn[0][1]:>7.4f}", flush=True)
        print()

        # Loss vs. Generation
        plt.figure(figsize=(10, 6))
        plt.title("Fitness vs. Generation")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        fitnesses = [result[0][1] for result in gen_results]
        plt.plot(range(self.NUM_GENS), fitnesses)
        plt.show()
        plt.close()

        # print and graph best model
        best_model = tourn[0][0]
        best_model.print_props()
        best_model.graph_model(datetime.now())
        self.model_eval(best_model, True)

        # Expected Planetary Path
        plt.figure(figsize=(10, 6))
        plt.title("Planetary Paths")
        plt.xlabel("Date")
        plt.ylabel("Longitude")
        plt.yticks(range(0,361,30))
        plt.scatter(self.datetimes, self.longitudes, s=10, c='g', label="Expected", alpha=.5)
        plt.scatter(self.datetimes, [best_model.predict_pos(dt)[2] for dt in self.datetimes], s=10, c='r', label="Actual", alpha=.5)
        plt.legend()
        plt.show()
        plt.close()