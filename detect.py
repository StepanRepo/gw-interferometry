#! venv/bin/python

"""
Gravitational Wave Pulsar Timing Array Correlation Detector
===========================================================

Implements correlation-based detection methods for GW signals in PTA data.

Key Features:
- Calculates correlation patterns between pulsar pairs
- Implements point-source and imaging detection methods
- Uses astropy units throughout for physical consistency
- Optimized with numpy vectorization for performance
"""

import numpy as np
import scipy as sp
import astropy.units as u
from typing import List, Union, Tuple
from dataclasses import dataclass

import matplotlib.pyplot as plt
from generate import Pulsar, GWSource

from myplot import *
set_tex()


class PTACorrelationDetector:
    """A detector that calculates correlation patterns between pulsars in a PTA.

    This class implements the mathematical framework for detecting gravitational waves
    by correlating signals from multiple pulsars in a timing array. The implementation
    follows the theoretical framework described in the accompanying paper.

    Attributes
    ----------
    pulsars : List[Pulsar]
        List of pulsars in the timing array
    _phases : astropy.Quantity
        Array of optimal phase differences for each pulsar pair (in radians)
    """


    def __init__(self, pulsars: List['Pulsar']):
        """Initialize the correlation detector with a set of pulsars.

        Parameters
        ----------
        pulsars : List[Pulsar]
            List of pulsar objects that form the timing array
        """

        self.pulsars = pulsars

        # Initialize phase differences for all pulsar pairs
        n_pul = len(pulsars)
        self._phases = np.zeros(n_pul * (n_pul+1) //2) * u.rad

    def pair_beam_pattern(self,
                                i: int,
                                j: int,
                                gw_source: GWSource,
                                phase_diff: u.Quantity = None) -> Union[float, np.ndarray]:
        """Calculate the correlation beam pattern for a specific pulsar pair.
        
        This implements calculating the response of a pulsar pair to a GW 
        source at a given sky position.
        
        Parameters
        ----------
        i, j : int
            Indices of the pulsar pair in the array
        gw_source : GWSource
            GW source parameters (can be scalar or array for multiple directions)
        phase_diff : astropy.Quantity, optional
            Phase difference Δφ between pulsars in radians
            
        Returns
        -------
        Union[float, np.ndarray]
            Correlation value(s) ρ for the given pulsar pair and GW source
        """

        # Convert inputs to numpy arrays
        theta = np.asarray(gw_source.theta.to(u.rad).value, dtype=np.float64)
        phi = np.asarray(gw_source.phi.to(u.rad).value, dtype=np.float64)

        # Angular frequency of GW and phase difference between pulsars
        omega = gw_source.frequency.to(u.rad/u.s).value
        delta_phi = phase_diff.to(u.rad).value

        # Get the two pulsars
        p1 = self.pulsars[i]
        p2 = self.pulsars[j]

        # Get unit vectors and angular separation
        p1_vec = p1.get_unit_vector()
        p2_vec = p2.get_unit_vector()

        # Calculate general antenna pattern components
        F1 = self._calculate_antenna_components(gw_source, p1_vec)
        F2 = self._calculate_antenna_components(gw_source, p2_vec)

        # Calculate distance terms ℓ (in seconds)
        omega_dir = gw_source.unit_vector

        dot1 = np.einsum('i,i...->...', p1_vec, omega_dir)  # p1·Ω
        dot2 = np.einsum('i,i...->...', p2_vec, omega_dir)  # p2·Ω

        ell1 = p1.distance.to(u.s).value * (1 + dot1)
        ell2 = p2.distance.to(u.s).value * (1 + dot2)
        angle_diff = omega * (ell1 - ell2) / 2

        # Calculate correlation components 
        S = 2 * np.sin(omega*ell1/2) * np.sin(omega*ell2/2)

        # Combine antenna patterns
        FF_plus = F1[0]*F2[0] + F1[1]*F2[1]  # F1+F2+ + F1×F2×
        FF_cross = F1[0]*F2[1] - F1[1]*F2[0]  # F1+F2× - F1×F2+

        # Calculate the two dynamical components of the correlation
        D1 = np.cos(angle_diff)*FF_plus + np.sin(angle_diff)*FF_cross
        D2 = np.cos(angle_diff)*FF_cross - np.sin(angle_diff)*FF_plus

        # Final correlation (equation 2.23)
        correlation = gw_source.strain**2 * S * (np.cos(delta_phi)*D1 + np.sin(delta_phi)*D2)

        return correlation

    def _calculate_antenna_components(self,
                                      gw_source: GWSource,
                                      p_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the antenna pattern components F+ and F× for a pulsar.
        
        Parameters
        ----------
        gw_source : GWSource
            GW source parameters including direction and polarization
        p_vec : np.ndarray
            Unit vector pointing to the pulsar
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The (F+, F×) antenna pattern components
        """
        # GW direction unit vector
        omega_dir = gw_source.unit_vector

        # Polarization tensors
        e_plus = gw_source.e_plus
        e_cross = gw_source.e_cross

        denom = 1 + np.einsum('i,i...->...', p_vec, omega_dir)

        # Antenna pattern calculation (general form)
        F_plus = 0.5 * np.einsum('i...,j...,ij...->...', p_vec, p_vec, e_plus) / denom
        F_cross = 0.5 * np.einsum('i...,j...,ij...->...', p_vec, p_vec, e_cross) / denom

        return F_plus, F_cross

    def _get_arg(self, 
                 i: int, 
                 j: int) -> int:
        """Calculate the linearized index in the _phases array for pulsar pair (i,j).

        Uses triangular numbering to store only unique pairs (i ≤ j).

        Parameters
        ----------
        i, j : int
            Indices of the pulsar pair

        Returns
        -------
        int
            Index in the _phases array for this pair
        """

        N = len(self.pulsars)
        return i*N + j - i*(i+1)//2

    def fix_phases(self, phases: np.ndarray):
        """Set the phase differences for all pulsar pairs.

        Parameters
        ----------
        phases : np.ndarray
            Array of phase differences in radians
        """

        self._phases = phases

    def phase(self,
              i: int,
              j: int) -> u.Quantity:
        """Get the optimal phase difference for a pulsar pair.

        Parameters
        ----------
        i, j : int
            Indices of the pulsar pair

        Returns
        -------
        astropy.Quantity
            Optimal phase difference in radians
        """

        arg = self._get_arg(i, j)

        return self._phases[arg]

    def observable_correlation(self,
                             i: int,
                             j: int,
                             dt: u.Quantity) -> np.float64:
        """Calculate observed correlations between two pulsars for a given time lag.

        Parameters
        ----------
        i, j : int
            Indices of the pulsar pair
        dt : astropy.Quantity 
            The time lags for correlation calculation

        Returns
        -------
        np.float64
            Correlation value for the given time lag
        """

        if not np.isscalar(dt.value):
            print("Correlation is only possible for a single time lag")

        psr1 = self.pulsars[i]
        psr2 = self.pulsars[j]

        # Convert MJDs to consistent units (days)
        mjd1 = psr1.mjd * u.day
        mjd2 = psr2.mjd * u.day

        # Get redshift data
        z1 = psr1.redshifts
        z2 = psr2.redshifts

        # Determine common time range
        t0 = mjd1.min()
        t1 = mjd1.max() - dt
        T = mjd1.max() - mjd1.min()

        # Create evaluation points (same for all time lags)
        t = np.linspace(t0, t1, len(mjd1))

        # Interpolate the redshift data to common time points
        int_z1 = np.interp(t, mjd1, z1)
        int_z2 = np.interp(t+dt, mjd2, z2)

        # Normalization factor
        denom = (T/ (T - dt)).to(u.dimensionless_unscaled)

         # Calculate and return the correlation
        return np.einsum("i...,i...->...", int_z1, int_z2) / denom


    def point_detector(self, 
                       gw_source: GWSource, 
                       n_phases: int = 100) -> None:
        """Determine optimal phase differences for maximum response to a GW source.
        
        This method finds the phase differences that maximize the correlation response
        to a GW source at a given position using a quadratic optimization
        
        Parameters
        ----------
        gw_source : GWSource
            The target GW source to point at (must be a single point)
        n_phases : int, optional
            Number of phase test points (default: 100)
            
        Raises
        ------
        AttributeError
            If gw_source contains multiple points or has no frequency
        """
        # Input validation
        if not np.isscalar(gw_source.frequency.value):
            raise AttributeError("GW Source must contain only one point: center of the FoV")
        if gw_source.frequency.value <= 0:
            raise AttributeError("GW Source frequency must be positive")
        
        n_pul = len(self.pulsars)
        optimal_phases = np.zeros(n_pul*(n_pul+1)//2) * u.rad

        omega = gw_source.frequency.to(u.rad/u.day)

        # For each pulsar pair, find the phase difference that maximizes correlation
        for i in range(n_pul):
            for j in range(i, n_pul):

                D1 = self.pair_beam_pattern(i, j, gw_source, 0 * u.rad)
                D2 = self.pair_beam_pattern(i, j, gw_source, np.pi/2 * u.rad)

                dphi = np.arctan(D2/D1) 
                dphi = np.mod(dphi, 2*np.pi)
                dphi = dphi * u.rad

                corr = self.observable_correlation(i, j, dphi/omega)

                if corr < 0:
                    dphi += np.pi * u.rad
                    corr = self.observable_correlation(i, j, dphi/omega)

                # Find the phase that gives maximum correlation
                arg = self._get_arg(i, j)
                optimal_phases[arg] = dphi

        self.fix_phases(optimal_phases)





    def image_point(self, 
                    grid: GWSource,
                    lam: np.float64 = 1e0) -> np.ndarray:
        """Reconstruct a sky image of GW sources using the PTA data.

        This implements the imaging method in the point-like source 
        approximation using Tikhonov regularization to solve the inverse problem.

        Parameters
        ----------
        grid : GWSource
            Grid of sky positions to evaluate (must be square)
        lam : float, optional
            Regularization parameter (default: 1.0)

        Returns
        -------
        np.ndarray
            Reconstructed sky image showing GW source locations
        """

        n, k = grid.phi.shape

        if n != k:
            raise AttributeError("Please provide a square grid")

        n_pul = len(self.pulsars)
        n_pair = n_pul*(n_pul+1)//2


        # Calculate pixel sizes in the grid
        dphi = grid.phi[0, 1] - grid.phi[0, 0]
        dtheta = grid.theta[1, 0] - grid.theta[0, 0]
        theta_c = np.mean(theta)
        dOmega = np.sin(theta_c) * dphi*dtheta

        # Initialize matrices for the linear system
        R = np.empty(shape = (n_pair, n*n))
        c = np.empty(shape = (n_pair))

        print("Calculationg point-like approximation")

        # Calculate beam patterns and correlations for all pulsar pairs
        for i in range(n_pul):
            for j in range(i, n_pul):

                print(f"\rBeam patterns found: {self._get_arg(i, j)+1} / {n_pul*(n_pul+1)//2}", end = "")

                # Calculate beam pattern for this pair over the grid
                beam = pta.pair_beam_pattern(i, j, grid, phase_diff = self.phase(i, j))
                
                # Get observed correlation for this pair
                corr = pta.observable_correlation(i, j, self.phase(i, j)/center.frequency)

                arg = self._get_arg(i, j)

                R[arg] = beam.ravel()
                c[arg] = corr 


        print("\nInverting matrix R^T*R")

        # Solve the regularized linear system
        inv = R.T @ R  + lam*np.eye(n*n)
        inv = sp.linalg.cho_factor(inv)
        inv = sp.linalg.cho_solve(inv, np.eye(n*n))

        # Calculate the solution vector and reshape into an image
        a = inv @ R.T @ c
        a = a.reshape((n, n))

        return a







if __name__ == "__main__":
    # Example usage with default parameters
    n = 100  # Grid size
    center = [60, 30] * u.deg  # Center of field of view
    n_pul = 60  # Number of pulsars to use

    # Field of view width
    width = [12, 12] * u.arcmin

    # Create a grid of sky positions to evaluates
    phi = center[0] + np.linspace(-width[0]/2, width[0]/2, n)
    theta = center[1] + np.linspace(-width[1]/2, width[1]/2, n)
    phi, theta = np.meshgrid(phi, theta)

    # Create GW source objects for the grid and center position
    grid = GWSource(theta = theta,
                    phi = phi,
                    frequency = 1e-8 * u.Hz,
                    strain = 1
                    )

    center = GWSource(theta = center[1], 
                      phi = center[0],
                      frequency = 1e-8 * u.Hz,
                      strain = 1)



    # Load pulsars and create detector
    pulsars = Pulsar.load_collection("pulsars")[:n_pul]
    pta = PTACorrelationDetector(pulsars)

    # Find optimal phases to poit the antenna
    pta.point_detector(center)

    # Reconstruct the sky image
    img = pta.image_point(grid, lam = 1e1)
    

    # Plot the results


    
    plt.figure(figsize = (8/2.54, 8/2.54))    # publication format
    # plt.figure(figsize = (4, 4)) # view format 

    plt.title(f"$N = {n_pul}$")
    plt.xlabel(r"arcmin")
    plt.ylabel(r"arcmin")

    plt.imshow(np.abs(img),
               origin = "lower",
               cmap = "hot",
               extent = [-width[0].to(u.arcmin).value/2,
                         width[0].to(u.arcmin).value/2,
                         -width[1].to(u.arcmin).value/2,
                         width[1].to(u.arcmin).value/2],
               )

    plt.colorbar()

    save_image("pta_image.pdf")







