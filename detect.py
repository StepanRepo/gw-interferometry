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
from matplotlib.ticker import FormatStrFormatter

from generate import Pulsar, GWSource
from pathlib import Path

from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from scipy import linalg

import copy

from myplot import *

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

        self.sinle_beams = None
        self.pair_beams = None
        self.correlations = None
        self.svd = None





    def pair_beam_pattern(self,
                         i: int,
                         j: int,
                         gw_source1: GWSource,
                         gw_source2: GWSource = None,
                         phase_diff: u.Quantity = None) -> Union[float, np.ndarray]:
        """Calculate the correlation beam pattern ρ(Ω₁, Ω₂) for a pulsar pair.

        This implements calculating the response of a pulsar pair to GW sources at
        given sky positions, following equation (2.31) from the paper.

        Parameters
        ----------
        i, j : int
            Indices of the pulsar pair in the array
        gw_source1 : GWSource
            First GW source parameters (direction Ω₁)
        phase_diff : astropy.Quantity, optional
            Phase difference Δφ between pulsars in radians
        gw_source2 : GWSource, optional
            Second GW source parameters (direction Ω₂). If None, uses gw_source1.

        Returns
        -------
        Union[float, np.ndarray]
            Correlation value(s) ρ(Ω₁, Ω₂) for the given pulsar pair
        """
        # Use gw_source1 for both if gw_source2 is not provided
        gw_source2 = gw_source1 if gw_source2 is None else gw_source2

        # Convert inputs to numpy arrays
        theta1 = np.asarray(gw_source1.theta.to(u.rad).value, dtype=np.float64)
        phi1 = np.asarray(gw_source1.phi.to(u.rad).value, dtype=np.float64)
        theta2 = np.asarray(gw_source2.theta.to(u.rad).value, dtype=np.float64)
        phi2 = np.asarray(gw_source2.phi.to(u.rad).value, dtype=np.float64)

        # Angular frequency of GW and phase difference between pulsars
        omega = gw_source1.frequency.to(u.rad/u.s).value
        if phase_diff is None:
            phase_diff = self.phase(i, j)
        delta_phi = phase_diff.to(u.rad).value

        # Get the two pulsars
        p1 = self.pulsars[i]
        p2 = self.pulsars[j]

        # Get unit vectors
        p1_vec = p1.get_unit_vector()
        p2_vec = p2.get_unit_vector()

        # Calculate antenna pattern components for both sources
        F1_plus, F1_cross = self._calculate_antenna_components(gw_source1, p1_vec)
        F2_plus, F2_cross = self._calculate_antenna_components(gw_source2, p2_vec)

        # Calculate distance terms ℓ (in seconds)
        omega_dir1 = gw_source1.unit_vector
        omega_dir2 = gw_source2.unit_vector

        dot1 = np.einsum('i,i...->...', p1_vec, omega_dir1)  # p1·Ω₁
        dot2 = np.einsum('i,i...->...', p2_vec, omega_dir2)  # p2·Ω₂

        ell1 = p1.distance.to(u.s).value * (1 + dot1)
        ell2 = p2.distance.to(u.s).value * (1 + dot2)

        # Calculate the static component S (equation 2.18)
        sin_term1 = np.sin(omega * ell1 / 2)
        sin_term2 = np.sin(omega * ell2 / 2)
        S = 2 * sin_term1 * sin_term2

        # Duration of the experiment
        #T = (p1.mjd.max() - p1.mjd.min())  # in days
        #delta_t = delta_phi / omega / 86400  # convert phase diff to days
        #S *= (T - delta_t)

        # Combine antenna patterns (equations 2.19-2.20)
        FF_plus = F1_plus * F2_plus + F1_cross * F2_cross  # F1+F2+ + F1×F2×
        FF_cross = F1_plus * F2_cross - F1_cross * F2_plus  # F1+F2× - F1×F2+

        # Calculate the angle difference term
        angle_diff = omega * (ell1 - ell2) / 2

        # Calculate the dynamical components D1 and D2 (equation 2.21)
        D1 = (np.cos(angle_diff) * FF_plus +
              np.sin(angle_diff) * FF_cross)
        D2 = (np.cos(angle_diff) * FF_cross -
              np.sin(angle_diff) * FF_plus)

        # Final correlation (equation 2.17)
        beam = S * (np.cos(delta_phi) * D1 + np.sin(delta_phi) * D2)

        return beam

    def single_beam_pattern(self,
                         i: int,
                         grid: GWSource) -> Union[float, np.ndarray]:

        """Calculate the correlation beam pattern ρ(Ω₁, Ω₂) for a pulsar pair.

        This implements calculating the response of a pulsar pair to GW sources at
        given sky positions, following equation (2.31) from the paper.

        Parameters
        ----------
        i, j : int
            Indices of the pulsar pair in the array
        grid : GWSource
            First GW source parameters (direction Ω₁)
        phase_diff : astropy.Quantity, optional
            Phase difference Δφ between pulsars in radians

        Returns
        -------
        Union[float, np.ndarray]
            Correlation value(s) ρ(Ω₁, Ω₂) for the given pulsar pair
        """

        # Convert inputs to numpy arrays
        theta = np.asarray(grid.theta.to(u.rad).value, dtype=np.float64)
        phi = np.asarray(grid.phi.to(u.rad).value, dtype=np.float64)

        # Angular frequency of GW and phase difference between pulsars
        omega = grid.frequency.to(u.rad/u.s).value

        # Get the two pulsars
        p = self.pulsars[i]

        # Get unit vectors
        p_vec = p.get_unit_vector()

        # Calculate antenna pattern components for both sources
        F_plus, F_cross = self._calculate_antenna_components(grid, p_vec)
        F = F_plus - 1j*F_cross

        # Calculate distance terms ℓ (in seconds)
        omega_dir = grid.unit_vector

        dot = np.einsum('i,i...->...', p_vec, omega_dir)  # p1·Ω₁
        ell = p.distance.to(u.s).value * (1 + dot)

        # Calculate the static component S (equation 2.18)
        sin_term = np.sin(omega * ell / 2)
        exp_term = np.exp(-1j * omega*ell/2)


        # Final correlation (equation 2.17)
        beam = np.sqrt(2) * sin_term  * exp_term * F

        return beam


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
                             dt: u.Quantity = 0*u.s) -> np.float64:
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

        # Integration factor
        denom = (t[1]-t[0]).to_value(u.d) / (T-dt).to_value(u.d)

         # Calculate and return the correlation
        return np.einsum("i...,i...->...", int_z1, int_z2) * denom


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

                D1 = self.pair_beam_pattern(i, j,
                                            gw_source,
                                            phase_diff = 0 * u.rad)
                D2 = self.pair_beam_pattern(i, j, gw_source,
                                            phase_diff = np.pi/2 * u.rad)

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



    def fill_beams(self,
                   grid: GWSource,
                   phase_diff: float = None):

        n, k = grid.phi.shape
        n_pul = len(self.pulsars)
        n_pair = n_pul*(n_pul+1)//2


        # Initialize matrices for the linear system
        B = np.empty(shape = (n_pair, n, n))
        c = np.empty(shape = (n_pair))

        self.single_beams = np.empty((n_pul, n, k), dtype = np.complex128)
        self.pair_beams = np.empty((n_pair, n, k))

        for i in range(n_pul):
            self.single_beams[i] = self.single_beam_pattern(i, grid)


        # Calculate beam patterns and correlations for all pulsar pairs
        for i in range(n_pul):
            for j in range(i, n_pul):
                arg = self._get_arg(i, j)

                print(f"\rBeam patterns found: {arg+1} / {n_pair}", end = "")

                self.pair_beams[arg] = (self.single_beams[i] * self.single_beams[j].conj()).real


        print("")

    def fill_correlations(self):

        n_pul = len(self.pulsars)
        n_pair = n_pul*(n_pul+1)//2

        self.correlations = np.empty(n_pair)

        for i in range(n_pul):
            for j in range(i, n_pul):
                arg = self._get_arg(i, j)

                print(f"\rCorrelations found: {arg+1} / {n_pair}", end = "")

                self.correlations[arg] = self.observable_correlation(i, j)

        print("")


    def image_point1(self,
                    grid: GWSource,
                    refine_beams: bool = True) -> np.ndarray:
        """Reconstruct a sky image of GW sources using the PTA data.

        This implements the imaging method in the point-like source

        Parameters
        ----------
        grid : GWSource
            Grid of sky positions to evaluate (must be square)

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
        #dphi = grid.phi[0, 1] - grid.phi[0, 0]
        #dtheta = grid.theta[1, 0] - grid.theta[0, 0]
        #solid_angle = np.sin(grid.theta) * dtheta * dphi
        #solid_angle = solid_angle.to_value(u.rad**2)


        # Initialize matrices for the linear system
        if refine_beams or self.pair_beams is None:
            self.fill_beams(grid)

        self.fill_correlations()
        R = self.pair_beams.reshape((n_pair, -1))


        #inv = 1/np.sum(R*R, axis = 0)
        #asq = (R.T @ self.correlations) * inv
        inv = np.linalg.inv(R.T @ R + 1e0 * np.eye(n*n))
        asq = inv @ R.T @ self.correlations

        a = np.sqrt(np.abs(asq)).reshape((n, n))

        return a



    def image_clean(self,
                   grid: GWSource,
                   gain: np.float64 = 1e-1,
                   n_iter: int = 50,
                   ) -> np.ndarray:


        phi = grid.phi
        theta = grid.theta
        omega = grid.frequency
        n, _ = phi.shape
        n_pul = len(self.pulsars)
        n_pair = n_pul * (n_pul + 1) // 2



        # Precompute correlations
        pta = copy.deepcopy(self)

        # Initialize the image
        img = np.zeros(shape = (n, n))
        DI = pta.image_point(grid, refine_beams = True)

        # CLEAN loop
        for i in range(n_iter):
            # Find new set of correlations and construct a dirty image
            DI = pta.image_point(grid, refine_beams = False)

            # Find amplitude and position of the next point in the image
            argmax1d = np.argmax(DI)
            argmax = np.unravel_index(argmax1d, DI.shape)
            amplitude = gain * DI[argmax]

            # Save this point
            img[argmax] += amplitude

            np.save(f"maps/dirty{i}", DI)
            np.save(f"maps/clean{i}", img)


            plt.figure(figsize = (4, 4)) # view format
            #plt.figure(figsize = a4(.6, .6/np.sqrt(2))) # paper format

            plt.title(f"Dirty Map {i}")
            plt.xlabel(r"pix")
            plt.ylabel(r"pix")

            plt.imshow(DI,
                       origin = "lower",
                       cmap = "hot",
                       )
            plt.colorbar()
            #plt.plot(argmax[1], argmax[0], "ko")


            # Remove the point from observations
            source = GWSource(theta = theta[argmax],
                              phi = phi[argmax],
                              frequency = omega,
                              strain = -amplitude)

            for psr in pta.pulsars:
                psr.add_redshift(source)

        return img





    def image_point(self,
                    grid: GWSource,
                    refine_beams: bool = True) -> np.ndarray:

        n, k = grid.phi.shape

        if n != k:
            raise AttributeError("Please provide a square grid")

        n_pul = len(self.pulsars)
        n_pair = n_pul*(n_pul+1)//2


        # Initialize matrices for the linear system
        if refine_beams or self.pair_beams is None:
            self.fill_beams(grid)
            R = self.pair_beams.reshape((n_pair, -1))
            U, S, Vh = linalg.svd(R, full_matrices = False)

            self.svd = U, S, Vh

        # Calculate pixel sizes in the grid
        #dphi = grid.phi[0, 1] - grid.phi[0, 0]
        #dtheta = grid.theta[1, 0] - grid.theta[0, 0]
        #solid_angle = np.sin(grid.theta) * dtheta * dphi
        #solid_angle = solid_angle.to_value(u.rad**2)

        self.fill_correlations()
        rho = self.correlations

        # Initialize matrices for the linear system
        n = grid.phi.shape[0]
        U, S, Vh = self.svd


        #bad = (S/S[0] < 1e-2)
        # k = np.argmax(bad)
        # if (k == 0): k = len(S)
        k = 2*len(self.pulsars)

        img = ((U.T @ rho)[:k] / S[:k]) @ Vh[:k, :]
        img = np.sqrt(np.abs(img))

        return img.reshape((n, n))






    def image_psf(self,
                  grid: GWSource,
                  crop: int = None,
                  ) -> (np.ndarray, np.ndarray):

        def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, const = 0):
            x, y = xy
            a = np.cos(theta)**2/(2*sigma_x**2) + np.sin(theta)**2/(2*sigma_y**2)
            b = -np.sin(2*theta)/(4*sigma_x**2) + np.sin(2*theta)/(4*sigma_y**2)
            c = np.sin(theta)**2/(2*sigma_x**2) + np.cos(theta)**2/(2*sigma_y**2)

            img =  amplitude * np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2)) + const
            return img.ravel()

        phi = grid.phi
        theta = grid.theta
        omega = grid.frequency

        n, _ = phi.shape
        nh = n//2

        n_pul = len(self.pulsars)
        n_pair = n_pul * (n_pul + 1)//2
        center = [theta[nh, nh], phi[nh, nh]]


        # Add a point source in observations
        source = GWSource(theta = center[0],
                          phi = center[1],
                          frequency = omega,
                          strain = 1)

        pulsarsnew = self.pulsars.copy()

        for i, psr in enumerate(pulsarsnew):
            psr.redshifts[:] = 0.0
            psr.add_redshift(source)

        ptanew = PTACorrelationDetector(pulsarsnew)
        a = ptanew.image_point(grid, refine_beams = True)

        if crop is not None:
            phi = phi[nh - crop : nh + crop, nh - crop : nh + crop]
            theta = theta[nh - crop : nh + crop, nh - crop : nh + crop]
            a_cr = a[nh - crop : nh + crop, nh - crop : nh + crop]


            n, _ = phi.shape
            nh = n//2
        else:
            a_cr = a



        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        X, Y = np.meshgrid(x, y)

        i0 = [1, .5, .5, 1e-2, 1e-2, 0, 0]
        popt, pcov = curve_fit(gaussian_2d, (X, Y), a_cr.ravel(), p0 = i0,
                               bounds = ([0, .4, .4, 0, 0, 0, -np.pi],
                                         [2, .6, .6, .3, .3, np.pi, 2]))


        x = np.linspace(0, 1, grid.phi.shape[0])
        y = np.linspace(0, 1, grid.phi.shape[0])
        X, Y = np.meshgrid(x, y)

        popt[-1] = 0
        popt[1] = .5
        popt[2] = .5
        popt[3] = popt[3] * n / grid.phi.shape[0]
        popt[4] = popt[4] * n / grid.phi.shape[0]

        img = gaussian_2d((X, Y), *popt).reshape(grid.phi.shape)

        return a, img / np.sum(img)







if __name__ == "__main__":
    # Example usage with default parameters
    n = 100  # Grid size
    center = [60, 30] * u.deg  # Center of field of view
    n_pul = 50  # Max number of pulsars to use

    # Field of view width
    width = [10, 10] * u.arcmin

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
    n_pul = min(n_pul, len(pulsars))
    pta = PTACorrelationDetector(pulsars)

    # Reconstruct the sky image
    img = pta.image_clean(grid, gain = 1, n_iter = 33)
    dirty_beam, psf = pta.image_psf(grid)
    img = convolve2d(img, psf, mode = "same") / np.max(psf)

    np.save(f"maps/conv", img)




    plt.figure(figsize = (4, 4)) # view format
    #plt.figure(figsize = a4(.6, .6/np.sqrt(2))) # paper format

    plt.title(f"Clean Map $N = {n_pul}$")
    plt.xlabel(r"arcmin")
    plt.ylabel(r"arcmin")
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('$%.0f$'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('$%.0f$'))

    extent = [-width[0].to(u.arcmin).value/2,
              width[0].to(u.arcmin).value/2,
              -width[1].to(u.arcmin).value/2,
              width[1].to(u.arcmin).value/2]

    plt.imshow(img,
               origin = "lower",
               cmap = "hot",
               extent = extent,
               )
    plt.colorbar()


#    # Plot PSF
#    shifted = np.roll(psf/psf.max(), shift = (-n//3, n//3), axis = (0, 1))
#
#    plt.contour(shifted,
#                levels = [.5],  # adjust number of contour levels as needed
#                colors = "white",  # or any color that contrasts well
#                linewidths = .7,
#                origin = "lower",
#                extent = extent,
#                )


    #save_eps("pta_image", tight = True)
    save_image("pta_image.pdf", tight = True)







