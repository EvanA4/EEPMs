from evolver import RGM_Evolver
from geomodel import RGM_Initializer, RandGeoModel
import matplotlib.pyplot as plt


def main():
    evo = RGM_Evolver()
    deferent_model, _ = evo.simulate(1)
    evo.simulate(2, deferent_model)
    # TODO directly measure average days between retrogrades for preds
    # TODO maybe combine step and dldt penalties for stage 2 before refactor?
    # TODO failing to compute guaranteed epicycle
    # TODO on steep orbit paths msqe ignores retrograde
    # ini = RGM_Initializer()
    # ini.stage1(evo.datetimes[0], evo.longitudes[0], evo.avg_av)
    # model = RandGeoModel(ini)
    # model.properties[RandGeoModel.IDX_ECCENTRICITY] = 0
    # model.deferent_center = (0,0)
    # model.print_props()
    # model.graph_model()
    
    # # Expected Planetary Path
    # plt.figure(figsize=(10, 6))
    # plt.title("Planetary Paths")
    # plt.xlabel("Date")
    # plt.ylabel("Longitude")
    # plt.yticks(range(0, 361, 30))
    # plt.scatter(evo.datetimes, evo.longitudes, s=10, c='g', label="Expected", alpha=.5)
    # plt.scatter(
    #     evo.datetimes, [model.predict_pos(dt)[2] for dt in evo.datetimes],
    #     s=10, c='r', label="Actual", alpha=.5
    # )
    # plt.legend()
    # plt.show()
    # plt.close()


if __name__ == "__main__":
    main()


'''
TODO evolve deferent first (eccentricity, eccentric angle, ed_av, radii=0, epicycle angle) with msqe
TODO evolve epicycle second (radii, planet angle, pe_av)
TODO maybe final pass over all parameters with very tiny mutation?
'''