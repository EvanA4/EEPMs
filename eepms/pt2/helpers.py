import math
from datetime import datetime


def get_cumu_longs(longs: list[float], is_radians=False):
    half_jump = math.pi if is_radians else 180
    full_jump = 2 * half_jump
    offset = 0
    cumu_longs = []
    for i in range(1, len(longs)):
        wrapped_down = longs[i] - longs[i - 1] < -half_jump
        wrapped_up = longs[i] - longs[i - 1] > half_jump
        if wrapped_down:
            offset += full_jump
        elif wrapped_up:
            offset -= full_jump
        cumu_longs.append(longs[i] + offset)
    return cumu_longs


def get_steps(cumu_longs: list[float]):
    total_retrostep = 0
    num_retrosteps = 0
    total_prostep = 0
    num_prosteps = 0
    for i in range(1, len(cumu_longs)):
        diff = cumu_longs[i] - cumu_longs[i - 1]
        if diff < 0:
            total_retrostep += diff
            num_retrosteps += 1
        if diff > 0:
            total_prostep += diff
            num_prosteps += 1
    return (
        -total_retrostep / num_retrosteps if num_retrosteps else 0,
        total_prostep / num_prosteps if num_prosteps else 0
    )


def get_retro_times(cumu_longs):
    is_in_retrograde = False
    tmp_retro_start = -1
    retro_times: list[tuple[int, int]] = []
    for i in range(2, len(cumu_longs)):
        first = cumu_longs[i - 1] - cumu_longs[i - 2]
        second = cumu_longs[i] - cumu_longs[i - 1]
        if first > 0 > second and not is_in_retrograde:
            is_in_retrograde = True
            tmp_retro_start = i
        if first < 0 < second and is_in_retrograde:
            is_in_retrograde = False
            retro_times.append((tmp_retro_start, i))
            tmp_retro_start = -1
    return retro_times


def get_retro_stats(retro_times, dts: list[datetime], cumu_longs: list[float]):
    avg_gap = 0. # average number of days between start of retrogrades
    for i in range(1, len(retro_times)):
        td = dts[retro_times[i][0]] - dts[retro_times[i - 1][0]]
        days = td.days + td.seconds / 86400
        avg_gap += days / (len(retro_times) - 1)

    avg_len = 0.  # average number of days between start and end of retrograde
    avg_height = 0.  # average difference in longitude from retrograde
    for i in range(len(retro_times)):
        td = dts[retro_times[i][1]] - dts[retro_times[i][0]]
        days = td.days + td.seconds / 86400
        avg_len += days / len(retro_times)
        avg_height += (cumu_longs[retro_times[i][0]] - cumu_longs[retro_times[i][1]]) / len(retro_times)
    return avg_gap, avg_len, avg_height


def min_long_diff(first: float, second: float, is_radians: bool):
    # assumes longitudes are in degrees
    half_circle = math.pi if is_radians else 180
    full_circle = 2*math.pi if is_radians else 360
    long_top = max(first, second)
    long_bottom = min(first, second)
    start_diff_frac = min(long_top - long_bottom, full_circle - long_top + long_bottom) / half_circle
    return start_diff_frac


def first_retro(retro_times: list[tuple[int, int]], dts: list[datetime]):
    if len(retro_times) == 0: return 0
    td = dts[retro_times[0][0]] - dts[0]
    return td.days + td.seconds / 86400