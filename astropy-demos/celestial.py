from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body_barycentric, get_body
import numpy as np

'''
According to Stellarium at 2026-03-04 21:39:21...

Mercury
    RA: 23h 14m 18.7s
    Dec: -00° 54' 54.8"
    Distance: 0.65 AU

Venus
    RA: 23h 55m 56.1s
    Dec: -01° 50' 39.1"
    Distance: 1.65 AU

Mars
    RA: 22h 17m 30.7s
    Dec: -11° 47' 27.8"
    Distance: 2.34 AU

Jupiter
    RA: 07h 05m 54.1s
    Dec: +22° 55' 30.5"
    Distance: 4.66 AU

Saturn
    RA: 00h 11m 31.3s
    Dec: -01° 04' 21.3"
    Distance: 10.44

Uranus
    RA: 03h 42m 25.2s
    Dec: +19° 30' 03.1"
    Distance: 19.74

Neptune
    RA: 00h 06m 27.6s
    Dec: -00° 43' 25.0"
    Distance: 30.83
'''


def get_celestial(oc, ec):
    diff = oc-ec
    r = np.linalg.norm(diff)
    ra = np.degrees(np.arctan2(diff[1], diff[0])) % 360
    dec = np.degrees(np.arcsin(diff[2] / r))
    return ra, dec, r
    

def to_hms(ra):
    total_hours = ra / 15.0
    hours = int(total_hours)
    minutes = int((total_hours - hours) * 60)
    seconds = (total_hours - hours - minutes/60) * 3600
    return hours, minutes, seconds


def to_dsa(dec):
    deg = int(dec)
    minutes = int((dec - deg) * 60)
    seconds = (dec - deg - minutes/60) * 3600
    return deg, minutes, seconds


def print_planet(name, celestial):
    print(planet)
    ra_hms = to_hms(celestial[0])
    dec_neg = celestial[1] < 0
    dec_dsa = to_dsa(np.abs(celestial[1]))
    print(f"\tRA: {ra_hms[0]:02d}h {ra_hms[1]:02d}m {ra_hms[2]:02.1f}s")
    print(f"\tDec: {"-" if dec_neg else "+"}{dec_dsa[0]:02d}° {dec_dsa[1]:02d}\' {dec_dsa[2]:02.1f}\"")
    print(f"\tDistance: {celestial[2]:.2f} AU")


def get_pos(name: str):
    t = Time("2026-03-04 21:39:21")
    cartrep = get_body_barycentric(name, t)
    jc = cartrep.get_xyz().to_value()
    cartrep = get_body_barycentric('earth', t)
    ec = cartrep.get_xyz().to_value()
    celestial = get_celestial(jc, ec)
    print_planet(name, celestial)


PLANETS = ['Mercury', 'Venus', 'Mars', "Jupiter", 'Saturn', 'Uranus', "Neptune"]
for planet in PLANETS:
    get_pos(planet)