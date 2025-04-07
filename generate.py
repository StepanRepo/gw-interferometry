#! venv/bin/python

"""
Gravitational Wave Pulsar Timing Array Simulator
===============================================

A complete implementation of GW signal simulation in PTA data following:
- Anholm et al. (2009) formalism
- Proper astropy units throughout
- Realistic parameter ranges
- Comprehensive docstrings

This module provides classes to simulate gravitational wave sources and pulsars
for pulsar timing array (PTA) analysis. It handles unit conversions automatically
and includes functionality to save/load data to HDF5 files.
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

    This class represents a gravitational wave source with configurable parameters
    including position, frequency, strain amplitude, and other physical properties.
    It automatically handles unit conversions and calculates polarization tensors.

    Parameters
    ----------
    theta : u.Quantity
        Polar angle of GW source [rad]
    phi : u.Quantity
        Azimuthal angle of GW source [rad]
    frequency : u.Quantity
        GW frequency [Hz] or angular frequency [rad/s]
    strain : float, optional
        Characteristic strain amplitude (dimensionless), default=1
    distance : u.Quantity, optional
        Luminosity distance to source, default=100 Mpc
    inclination : u.Quantity, optional
        Inclination angle of binary orbit, default=0 rad
    phase : u.Quantity, optional
        Initial GW phase, default=0 rad

    Attributes
    ----------
    theta : u.Quantity
        Polar angle of GW source [rad]
    phi : u.Quantity
        Azimuthal angle of GW source [rad]
    frequency : u.Quantity
        Angular frequency of GW [rad/s]
    strain : float
        Characteristic strain amplitude (dimensionless)
    distance : u.Quantity
        Luminosity distance to source [yr]
    inclination : u.Quantity
        Inclination angle of binary orbit [rad]
    phase : u.Quantity
        Initial GW phase [rad]
    e_plus : ndarray
        Plus polarization tensor
    e_cross : ndarray
        Cross polarization tensor
    unit_vector : ndarray
        Unit vector pointing to GW source

    Example
    -------
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
        """Validate and convert units on initialization.

        Ensures all quantities have appropriate units and calculates
        derived properties like polarization tensors.
        """

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



        m = np.array([np.sin(self.phi), -np.cos(self.phi), np.zeros(self.phi.shape)])
        n = np.array([
            np.cos(self.theta)*np.cos(self.phi),
            np.cos(self.theta)*np.sin(self.phi),
            -np.sin(self.theta)
            ])

        self.e_plus = np.einsum('i...,j...->ij...', m, m) - np.einsum('i...,j...->ij...', n, n)
        self.e_cross = np.einsum('i...,j...->ij...', m, n) + np.einsum('i...,j...->ij...', n, m)

        self.unit_vector  = np.array([
            np.sin(self.theta) * np.cos(self.phi),
            np.sin(self.theta) * np.sin(self.phi),
            np.cos(self.theta)
            ])



class Pulsar:
    """A pulsar in the timing array.

    This class represents a pulsar with its astrometric parameters and timing data.
    It includes methods to calculate GW-induced redshift, generate observation times,
    and save/load data to/from HDF5 files.

    Parameters
    ----------
    ra : u.Quantity
        Right ascension [rad]
    dec : u.Quantity
        Declination [rad]
    distance : u.Quantity
        Distance to pulsar [kpc or yr]
    name : str, optional
        Pulsar identifier, default=""

    Attributes
    ----------
    ra : u.Quantity
        Right ascension [rad]
    dec : u.Quantity
        Declination [rad]
    distance : u.Quantity
        Distance to pulsar [yr]
    name : str
        Pulsar identifier
    mjd : ndarray or None
        Array of observation times
    redshifts : ndarray or None
        Induced redshift values
    """

    def __init__(self, ra: u.Quantity, dec: u.Quantity,
                 distance: u.Quantity, name: str = ""):
        """Initialize a pulsar with sky position and distance.

        Parameters
        ----------
        ra : u.Quantity
            Right ascension (0 to 2π rad)
        dec : u.Quantity
            Declination (-π/2 to π/2 rad)
        distance : u.Quantity
            Distance to pulsar [kpc or light-years]
        name : str, optional
            Pulsar identifier, default=""
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
        """Generate default name from coordinates.

        Returns
        -------
        str
            Pulsar name in J2000 format (e.g., J0534+2200)
        """

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

        Parameters
        ----------
        n : int, optional
            Number of pulsars to generate, default=1
        min_distance : u.Quantity, optional
            Minimum pulsar distance, default=0.3 kpc
        max_distance : u.Quantity, optional
            Maximum pulsar distance, default=5 kpc

        Returns
        -------
        List[Pulsar]
            List of randomly generated Pulsar objects
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

        Returns
        -------
        ndarray
            3D unit vector in Cartesian coordinates
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

        Parameters
        ----------
        start_mjd : float, optional
            Starting Modified Julian Date, default=58000.0
        duration : u.Quantity, optional
            Total observation duration, default=10 yr
        cadence : u.Quantity, optional
            Time between observations, default=14 days

        Returns
        -------
        ndarray
            Array of MJD observation times
        """

        # Calculate number of observations based on duration and cadence
        n_obs = int((duration / cadence).to(u.dimensionless_unscaled))

        # Generate evenly spaced observation times
        self.mjd = start_mjd + np.arange(n_obs) * (cadence.to(u.day).value)
        self.redshifts = np.zeros(n_obs, dtype = np.float64)

        return self.mjd


    def add_redshift(self, gw_source: GWSource) -> np.ndarray:
        """Calculate GW-induced redshift in pulsar signal.

        Implements the Anholm et al. (2009) formalism for calculating
        the redshift induced by a gravitational wave.

        Parameters
        ----------
        gw_source : GWSource
            GW source parameters

        Returns
        -------
        ndarray
            Array of redshift values at each observation time
        """

        t = self.mjd * u.day

        # GW propagation direction
        gw_dir = gw_source.unit_vector
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

        # Add to existing redshifts (allows multiple sources)
        self.redshifts += redshift

        return redshift

    def save_to_file(self, filename: str):
        """Save pulsar data to individual HDF5 file.

        Parameters
        ----------
        filename : str
            Output file path (.h5 recommended)
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

        Parameters
        ----------
        filename : str
            Input file path

        Returns
        -------
        
        Pulsar
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

        Parameters
        ----------

        pulsars: List[Pulsar]
            List of Pulsar objects
        directory: str
            Target directory path
        """


        Path(directory).mkdir(parents=True, exist_ok=True)

        for pulsar in pulsars:
            pulsar.save_to_file(f"{directory}/{pulsar.name}.h5")

    @classmethod
    def load_collection(cls, directory: str) -> List['Pulsar']:
        """Load multiple pulsars from a directory of files.

        Parameters
        ----------
        directory: str
            Source directory path

        Returns
        -------
        
        List[Pulsar]
            List of reconstructed Pulsar objects
        """


        pulsars = []
        for filepath in Path(directory).glob('*.h5'):
            pulsars.append(cls.load_from_file(str(filepath)))
        return pulsars



    def __str__(self) -> str:
        """Return formatted string representation of the class.

        Returns
        -------
        
        str
            Formatted string with the class' parameters
        """

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


    # Define a GW source
    source1 = GWSource(theta = 30*u.deg,
                      phi = 60*u.deg,
                      frequency = 1e-8 * u.Hz,
                      strain = 1)


    # Define a second GW source to add it to the observations
    source2 = GWSource(theta = 30*u.deg,
                      phi = 60*u.deg + 3*u.arcmin, 
                      frequency = 1e-8 * u.Hz,
                      strain = 1)

    # Generate a set of 60 pulsars
    pulsars = Pulsar.generate_random(60)

    # Add influence of defined sources to the set of pulsars
    for psr in pulsars:
        psr.generate_observation_times()
        psr.add_redshift(source1)
        #psr.add_redshift(source2)

    # Save simulated data 
    Pulsar.save_collection(pulsars, "pulsars")





