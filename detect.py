#! venv/bin/python

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
    """Calculates correlation patterns from the paper.

    Attributes:
        pulsars (List[Pulsar]): List of pulsars in the array
    """

    def __init__(self, pulsars: List['Pulsar']):
        n_pul = len(pulsars)

        self.pulsars = pulsars
        self._phases = np.zeros(n_pul * (n_pul+1) //2) * u.rad

    def pair_beam_pattern(self,
                                i: int,
                                j: int,
                                gw_source: GWSource,
                                phase_diff: u.Quantity = None) -> Union[float, np.ndarray]:
        """Calculate correlation for specific pulsar pair.

        Args:
            i, j: Indices of pulsar pair
            gw_source: GW source parameters (scalar or array)
            phase_diff: Phase difference Δφ [rad]

        Returns:
            Correlation value(s) ρ following equation (2.22)
        """

        # Convert inputs to numpy arrays
        theta = np.asarray(gw_source.theta.to(u.rad).value, dtype=np.float64)
        phi = np.asarray(gw_source.phi.to(u.rad).value, dtype=np.float64)

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

        dot1 = np.einsum('i,i...->...', p1_vec, omega_dir)
        dot2 = np.einsum('i,i...->...', p2_vec, omega_dir)

        ell1 = p1.distance.to(u.s).value * (1 + dot1)
        ell2 = p2.distance.to(u.s).value * (1 + dot2)

        # Calculate correlation components (equations 2.24-2.26)
        B = 2 * gw_source.strain**2  \
                * np.sin(omega*ell1/2) * np.sin(omega*ell2/2)
        angle_diff = omega * (ell1 - ell2) / 2

        # Combine antenna patterns
        FF_plus = F1[0]*F2[0] + F1[1]*F2[1]  # F1+F2+ + F1×F2×
        FF_cross = F1[0]*F2[1] - F1[1]*F2[0]  # F1+F2× - F1×F2+

        rho1 = B * (np.cos(angle_diff)*FF_plus + np.sin(angle_diff)*FF_cross)
        rho2 = B * (np.cos(angle_diff)*FF_cross - np.sin(angle_diff)*FF_plus)

        # Final correlation (equation 2.23)
        correlation = np.cos(delta_phi)*rho1 + np.sin(delta_phi)*rho2

        return correlation
        #return correlation[0] if np.isscalar(gw_source.theta.value) else correlation

    def _calculate_antenna_components(self,
                                      gw_source: GWSource,
                                      p_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate general antenna pattern components F+ and F×.

        Args:
            phi: Azimuthal angle(s) [rad]
            theta: Polar angle(s) [rad]
            p_vec: Pulsar unit vector

        Returns:
            Tuple of (F+, F×) arrays
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
        N = len(self.pulsars)
        return i*N + j - i*(i+1)//2

    def fix_phases(self, phases: np.ndarray):
        self._phases = phases

    def phase(self,
              i: int,
              j: int) -> u.Quantity:

        arg = self._get_arg(i, j)
        return self._phases[arg]

    def observable_correlation(self,
                               i: int, 
                               j: int, 
                               dt: u.Quantity = None):
        psr1 = self.pulsars[i]
        psr2 = self.pulsars[j]

        mjd1 = psr1.mjd * u.day
        mjd2 = psr2.mjd * u.day

        z1 = psr1.redshifts
        z2 = psr2.redshifts

        t0 = mjd1.min()
        t1 = mjd1.max() - dt

        T = mjd1.max() - mjd1.min()

        t = np.linspace(t0, t1, len(mjd1))

        int_z1 = np.interp(t, mjd1, z1)
        int_z2 = np.interp(t+dt, mjd2, z2)

        denom = (T/ (T - dt)).to(u.dimensionless_unscaled)

        return np.einsum("i...,i...->...", int_z1, int_z2) / denom



    def point_detector(self, gw_source: GWSource):
        n_pul = len(self.pulsars)
        pp = np.linspace(0, 2*np.pi, 100) * u.rad
        c = np.empty_like(pp)


        if not np.isscalar(gw_source.frequency.value):
            raise AttributeError("GW Source must contain only one point: center of the FoV")

        for i in range(n_pul):
            for j in range(i, n_pul):

                c = self.observable_correlation(i, j,
                                               pp/gw_source.frequency)

                idx = np.argmax(c)
                arg = self._get_arg(i, j)

                self._phases[arg] = pp[idx]

    def image_point(self, 
                    grid: GWSource,
                    lam: np.float64 = 1e0) -> np.ndarray:

        n, k = grid.phi.shape

        dphi = grid.phi[0, 1] - grid.phi[0, 0]
        dtheta = grid.theta[1, 0] - grid.theta[0, 0]
        theta_c = np.mean(theta)
        dOmega = np.sin(theta_c) * dphi*dtheta

        if n != k:
            raise AttributeError("Please provide a square grid")

        n_pul = len(self.pulsars)
        n_pair = n_pul*(n_pul+1)//2

        R = np.empty(shape = (n_pair, n*n))
        c = np.empty(shape = (n_pair))

        print("Calculationg point-like approximation")

        for i in range(n_pul):
            for j in range(i, n_pul):

                print(f"\rBeam patterns found: {self._get_arg(i, j)+1} / {n_pul*(n_pul+1)//2}", end = "")

                beam = pta.pair_beam_pattern(i, j, 
                                             grid, 
                                             phase_diff = self.phase(i, j))

                corr = pta.observable_correlation(i, j,
                                                  self.phase(i, j)/center.frequency)

                arg = self._get_arg(i, j)

                R[arg] = beam.ravel()
                c[arg] = corr 


        print("")
        print("Inverting matrix R^T*R")

        inv = R.T @ R  + lam*np.eye(n*n)
        inv = sp.linalg.cho_factor(inv)
        inv = sp.linalg.cho_solve(inv, np.eye(n*n))

        a = inv @ R.T @ c
        a = a.reshape((n, n))

        return np.abs(a)







if __name__ == "__main__":

    n = 100
    center = [60, 30] * u.deg
    n_pul = 60


    width = [12, 12] * u.arcmin
    ones = np.ones(shape = (n, n))



    # make a grid of explored pixels
    phi = center[0] + np.linspace(-width[0]/2, width[0]/2, n)
    theta = center[1] + np.linspace(-width[1]/2, width[1]/2, n)

    phi, theta = np.meshgrid(phi, theta)

    grid = GWSource(theta = theta,
                    phi = phi,
                    frequency = 1e-8 * u.Hz,
                    strain = 1
                    )

    center = GWSource(theta = center[1], 
                      phi = center[0],
                      frequency = 1e-8 * u.Hz,
                      strain = 1)



    pulsars = Pulsar.load_collection("pulsars")[:n_pul]
    pta = PTACorrelationDetector(pulsars)

    # Find apropriate phases to poit the antenna
    pta.point_detector(center)

    img = pta.image_point(grid, lam = 1e1)
    

    plt.figure(figsize = (8/2.54, 8/2.54))
    plt.title(f"$N = {n_pul}$")
    plt.xlabel(r"arcmin")
    plt.ylabel(r"arcmin")
    plt.xticks([-6, -3, 0, 3, 6])
    plt.yticks([-6, -3, 0, 3, 6])

    plt.imshow(np.abs(img),
               origin = "lower",
               cmap = "hot",
               extent = [-width[0].to(u.arcmin).value/2,
                         width[0].to(u.arcmin).value/2,
                         -width[1].to(u.arcmin).value/2,
                         width[1].to(u.arcmin).value/2],
               #vmin = 0, vmax = 1
               )

    #plt.plot(n/2, n/2, "*")
    #plt.colorbar()

    save_image("123.pdf", tight = True)







