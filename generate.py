#! venv/bin/python

"""
Gravitational Wave Pulsar Timing Array Simulator
===============================================

A complete implementation of GW signal simulation in PTA data following:
- Anholm et al. (2009) formalism
- Proper astropy units throughout
- Realistic parameter ranges
- Comprehensive docstrings
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from typing import Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import Angle

from myplot import *
set_tex()



@dataclass
class GWSource:
    """A gravitational wave source for PTA simulations.

    Attributes:
        theta (u.Quantity): Polar angle of GW source [rad]
        phi (u.Quantity): Azimuthal angle of GW source [rad]
        frequency (u.Quantity): GW frequency [Hz]
        strain (float): Characteristic strain amplitude (dimensionless)
        distance (u.Quantity): Luminosity distance to source [Mpc]
        inclination (u.Quantity): Inclination angle of binary orbit [rad]
        phase (u.Quantity): Initial GW phase [rad]

    Example:
        >>> gw = GWSource(theta=45*u.deg, phi=30*u.deg,
        ...              frequency=8*u.nanohertz, strain=1e-14)
    """
    theta: u.Quantity
    phi: u.Quantity
    frequency: u.Quantity
    strain: float = 1
    distance: u.Quantity = 100 * u.Mpc
    inclination: u.Quantity = 0 * u.rad
    phase: u.Quantity = 0 * u.rad


    def __post_init__(self):
        """Validate and convert units on initialization."""
        self.theta = self.theta.to(u.rad)
        self.phi = self.phi.to(u.rad)
        self.inclination = self.inclination.to(u.rad)
        self.phase = self.phase.to(u.rad)

        if self.frequency.unit.physical_type == "frequency":
            self.frequency = 2*np.pi * self.frequency.to(u.Hz).value * u.rad/u.s
        else:
            self.frequency = self.frequency

        if self.distance.unit.physical_type == "length":
            self.distance = (self.distance / const.c).to(u.yr)


    def get_direction_vector(self) -> np.ndarray:
        """Get unit vector in GW propagation direction.

        Returns:
            np.ndarray: 3D unit vector in Cartesian coordinates
        """
        return np.array([
            np.sin(self.theta.value) * np.cos(self.phi.value),
            np.sin(self.theta.value) * np.sin(self.phi.value),
            np.cos(self.theta.value)
        ])

class Pulsar:
    """A pulsar in the timing array.

    Attributes:
        ra (u.Quantity): Right ascension [rad]
        dec (u.Quantity): Declination [rad]
        distance (u.Quantity): Distance to pulsar [kpc]
        name (str): Pulsar identifier
        psi (u.Quantity): Polarization angle [rad]
        mjd (np.ndarray): Array of observation times
        redshifts (np.ndarray): Induced redshift values
    """

    def __init__(self, ra: u.Quantity, dec: u.Quantity,
                 distance: u.Quantity, name: str = ""):
        """Initialize a pulsar with sky position and distance.

        Args:
            ra: Right ascension (0 to 2π rad)
            dec: Declination (-π/2 to π/2 rad)
            distance: Distance to pulsar [kpc]
            name: Pulsar identifier (default "")
            psi: Polarization angle (0 to π rad, random if None)
        """

        self.ra = Angle(ra.to(u.rad))
        self.dec = Angle(dec.to(u.rad))
        self.name = name or self._generate_default_name()
        self.mjd = None
        self.redshifts = None

        if distance.unit.physical_type == "time":
            self.distance = distance.to(u.yr)
        elif distance.unit.physical_type == "length":
            self.distance = (distance / const.c).to(u.yr)
        else:
            raise AttributeError("Wrong physical unit for distance")
    
    def _generate_default_name(self) -> str:
        """Generate default name from coordinates."""

        ra_hms = self.ra.to_string(unit = "hour", 
                                   sep = "", 
                                   decimal = False, 
                                   pad = True, 
                                   fields = 2)

        dec_dms = self.dec.to_string(unit = "deg", 
                                     sep = "", 
                                     decimal = False, 
                                     pad = True, 
                                     fields = 2,
                                     alwayssign = True)

        return f"J{ra_hms}{dec_dms}"

    @classmethod
    def generate_random(cls, n: int = 1,
                      min_distance: u.Quantity = 0.3 * u.kpc,
                      max_distance: u.Quantity = 5 * u.kpc) -> List['Pulsar']:
        """Generate pulsars with realistic Galactic distribution.

        Args:
            n: Number of pulsars to generate
            min_distance: Minimum pulsar distance [kpc]
            max_distance: Maximum pulsar distance [kpc]

        Returns:
            List of Pulsar objects with random positions
        """
        ra = 2 * np.pi * np.random.rand(n) * u.rad
        dec = np.arcsin(2 * np.random.rand(n) - 1) * u.rad
        distances = min_distance + (max_distance - min_distance) * np.random.rand(n)
        
        if n == 1:
            return cls(ra[i], dec[i], distances[i])
        else:
            return [cls(ra[i], dec[i], distances[i]) for i in range(n)]

    def get_unit_vector(self) -> np.ndarray:
        """Get unit vector pointing to pulsar.

        Returns:
            np.ndarray: 3D unit vector in Cartesian coordinates
        """
        return np.array([
            np.cos(self.ra.value) * np.cos(self.dec.value),
            np.sin(self.ra.value) * np.cos(self.dec.value),
            np.sin(self.dec.value)
        ])


    def generate_observation_times(self, start_mjd: float = 58000.0, 
                                   duration: u.Quantity = 10 * u.yr,
                                   cadence: u.Quantity = 14.0 * u.day) -> np.ndarray:
        """Generate realistic observation schedule.

        Returns:
            Array of MJD observation times
        """
        n_obs = int((duration / cadence).to(u.dimensionless_unscaled))
        self.mjd = start_mjd + np.arange(n_obs) * (cadence.to(u.day).value)
        return self.mjd


    def calculate_redshift(self, gw_source: GWSource) -> np.ndarray:
        """Calculate GW-induced redshift in pulsar signal.

        Args:
            gw_source: GW source parameters

        Returns:
            Array of redshift values at each observation time
        """
        t = self.mjd * u.day

        # GW propagation direction
        gw_dir = gw_source.get_direction_vector()
        p = self.get_unit_vector()
        mu = np.dot(gw_dir, p)

        # Polarization basis vectors
        m = np.array([np.sin(gw_source.phi.value), -np.cos(gw_source.phi.value), 0])
        n = np.array([
            np.cos(gw_source.theta.value) * np.cos(gw_source.phi.value),
            np.cos(gw_source.theta.value) * np.sin(gw_source.phi.value),
            -np.sin(gw_source.theta.value)
        ])

        # Polarization tensors
        e_plus = np.outer(m, m) - np.outer(n, n)
        e_cross = np.outer(m, n) + np.outer(n, m)

        # Antenna patterns
        term = 0.5 * (np.outer(p, p)) / (1 + mu)
        F_plus = np.sum(term * e_plus)
        F_cross = np.sum(term * e_cross)


        # Earth and pulsar terms
        phase_earth = (gw_source.frequency * t + gw_source.phase)
        phase_pulsar = (gw_source.frequency * (t - self.distance * (1 + mu)) + gw_source.phase)

        # Metric perturbation difference
        h_plus_e = gw_source.strain * np.cos(phase_earth)
        h_plus_p = gw_source.strain * np.cos(phase_pulsar)
        h_cross_e = gw_source.strain * np.sin(phase_earth)
        h_cross_p = gw_source.strain * np.sin(phase_pulsar)

        delta_h_plus = h_plus_p - h_plus_e
        delta_h_cross = h_cross_p - h_cross_e

        # Redshift calculation
        redshift = delta_h_plus * F_plus + delta_h_cross * F_cross

        # Store results
        self.redshifts = redshift

        return redshift

    def save_to_file(self, filename: str):
        """Save pulsar data to individual HDF5 file.
        
        Args:
            filename: Output file path (.h5 recommended)
        """
        with h5py.File(filename, 'w') as f:
            # Store astrometric parameters
            f.attrs['ra'] = self.ra.value
            f.attrs['dec'] = self.dec.value
            f.attrs['distance'] = self.distance.value
            f.attrs['name'] = self.name
            f.attrs['ra_unit'] = str(self.ra.unit)
            f.attrs['dec_unit'] = str(self.dec.unit)
            f.attrs['distance_unit'] = str(self.distance.unit)
            
            # Store time series if available
            if self.mjd is not None:
                f.create_dataset('mjd', data=self.mjd, compression='gzip')
            if self.redshifts is not None:
                f.create_dataset('redshifts', data=self.redshifts, compression='gzip')

    @classmethod
    def load_from_file(cls, filename: str) -> 'Pulsar':
        """Load pulsar from individual HDF5 file.
        
        Args:
            filename: Input file path
            
        Returns:
            Reconstructed Pulsar object
        """
        with h5py.File(filename, 'r') as f:
            # Reconstruct with proper units
            pulsar = cls(
                ra=f.attrs['ra'] * u.Unit(f.attrs['ra_unit']),
                dec=f.attrs['dec'] * u.Unit(f.attrs['dec_unit']),
                distance=f.attrs['distance'] * u.Unit(f.attrs['distance_unit']),
                name=f.attrs['name']
            )
            
            # Load time series if available
            if 'mjd' in f:
                pulsar.mjd = f['mjd'][:]
            if 'redshifts' in f:
                pulsar.redshifts = f['redshifts'][:]
                
        return pulsar

    @classmethod
    def save_collection(cls, pulsars: List['Pulsar'], directory: str):
        """Save multiple pulsars to individual files in a directory.
        
        Args:
            pulsars: List of Pulsar objects
            directory: Target directory path
        """
        Path(directory).mkdir(parents=True, exist_ok=True)

        for pulsar in pulsars:
            pulsar.save_to_file(f"{directory}/{pulsar.name}.h5")

    @classmethod
    def load_collection(cls, directory: str) -> List['Pulsar']:
        """Load multiple pulsars from a directory of files.
        
        Args:
            directory: Source directory path
            
        Returns:
            List of reconstructed Pulsar objects
        """
        pulsars = []
        for filepath in Path(directory).glob('*.h5'):
            pulsars.append(cls.load_from_file(str(filepath)))
        return pulsars



    def __str__(self) -> str:
        """Return formatted string representation with c=1 units."""
        # Coordinate conversion

        ra_hms = self.ra.to_string(unit = "hour", 
                                   sep = ":", 
                                   decimal = False, 
                                   pad = True, 
                                   precision = 2)

        dec_dms = self.dec.to_string(unit = "deg", 
                                     sep = ":", 
                                     decimal = False, 
                                     pad = True, 
                                     alwayssign = True,
                                     precision = 2)
        
        # Distance formatting
        if self.distance < 1 * u.kyr:
            dist_str = f"{self.distance.to(u.year):.3f}"
        else:
            dist_str = f"{self.distance.to(u.kyr):.3f}"
        
        # Build output
        header = f"Pulsar {self.name}" if self.name else "Unnamed Pulsar"
        coords = f"Coordinates: RA {ra_hms} | Dec {dec_dms}"
        distance = f"Distance:    {dist_str}"
        
        # Observation info
        obs_info = ""
        if self.mjd is not None:
            obs_duration = (self.mjd.max() - self.mjd.min()) * u.day
            obs_info = (f"\nObservations: {len(self.mjd)} points "
                       f"over {obs_duration.to(u.year):.2f}")
        
        return (f"=== {header} ===\n"
                f"{coords}\n"
                f"{distance}"
                f"{obs_info}")




# Example usage with realistic values
if __name__ == '__main__':

    np.random.seed(42)


    source = GWSource(theta = 30*u.deg, 
                      phi = 60*u.deg,
                      frequency = 1e-8 * u.Hz,
                      strain = 1)

    pulsars = Pulsar.generate_random(60)

    for psr in pulsars:
        psr.generate_observation_times()
        psr.calculate_redshift(source)

    Pulsar.save_collection(pulsars, "pulsars")
    pulsars = Pulsar.load_collection("pulsars")




