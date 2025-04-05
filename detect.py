#! venv/bin/python

import numpy as np
import astropy.units as u
from typing import List, Union, Tuple
from dataclasses import dataclass

import matplotlib.pyplot as plt
from generate import Pulsar, GWSource

from myplot import *
set_tex()


@dataclass
class PTACorrelationDetector:
    """Calculates correlation patterns from the paper.

    Attributes:
        pulsars (List[Pulsar]): List of pulsars in the array
    """

    pulsars: List['Pulsar']

    def pair_beam_pattern(self,
                                i: int,
                                j: int,
                                gw_source: GWSource,
                                phase_diff: u.Quantity = 0*u.rad) -> Union[float, np.ndarray]:
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
        gamma = np.arccos(np.clip(np.dot(p1_vec, p2_vec), -1, 1))

        # Calculate general antenna pattern components
        F1 = self._calculate_antenna_components(phi, theta, p1_vec)
        F2 = self._calculate_antenna_components(phi, theta, p2_vec)

        # Calculate distance terms ℓ (in seconds)
        omega_dir = gw_source.get_direction_vector()

        dot1 = np.einsum('i,i...->...', p1_vec, omega_dir)
        dot2 = np.einsum('i,i...->...', p2_vec, omega_dir)

        ell1 = p1.distance.to(u.s).value * (1 + dot1)
        ell2 = p2.distance.to(u.s).value * (1 + dot2)

        # Calculate correlation components (equations 2.24-2.26)
        B = 2 * gw_source.strain**2 * np.sin(omega*ell1/2) * np.sin(omega*ell2/2)
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
                                   phi: np.ndarray,
                                   theta: np.ndarray,
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
        omega_dir = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

        # Polarization basis vectors
        m = np.array([np.sin(phi), -np.cos(phi), np.zeros_like(phi)])
        n = np.array([
            np.cos(theta)*np.cos(phi),
            np.cos(theta)*np.sin(phi),
            -np.sin(theta)
        ])

        # Polarization tensors
        e_plus = np.einsum('i...,j...->ij...', m, m) - np.einsum('i...,j...->ij...', n, n)
        e_cross = np.einsum('i...,j...->ij...', m, n) + np.einsum('i...,j...->ij...', n, m)

        denom = 1 + np.einsum('i,i...->...', p_vec, omega_dir)


        # Antenna pattern calculation (general form)
        F_plus = 0.5 * np.einsum('i...,j...,ij...->...', p_vec, p_vec, e_plus) / denom
        F_cross = 0.5 * np.einsum('i...,j...,ij...->...', p_vec, p_vec, e_cross) / denom

        return F_plus, F_cross


def plot(f, *args, **kwargs):
    n, m = f.shape

    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"},
                           figsize = (12,8))

    #ax.imshow(ff1, extent = [-180, 180, -90, 90])
    lon = np.linspace(-np.pi, np.pi, n)
    lat = np.linspace(-np.pi/2., np.pi/2., m)
    Lon,Lat = np.meshgrid(lon,lat)

    img = ax.pcolormesh(Lon, Lat, f, shading = "nearest",
                  cmap = "hot",
                  linewidth = 0, rasterized=True,
                        *args, **kwargs)
    ax.set_xlabel(r"$\varphi$")
    ax.set_ylabel(r"$\theta$")
    plt.colorbar(img, ax=ax, orientation='horizontal', shrink=0.6)


    return fig, ax

def corr(psr1: Pulsar, 
         psr2: Pulsar, 
         dt: np.float64 = 0):

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



def get_arg(i, j, N):
    return i*N + j - i*(i+1)//2
    

def grad_hess(corrs, a, beams, lam):

    n = len(a)

    grad = np.zeros_like(a)
    hess = np.zeros(shape = (n, n))
    I = np.ones_like(a)


    for i, rho in enumerate(beams):

        rho_a = rho @ a

        residual = (corrs[i] - a.T @ rho_a) 

        grad += -4 * residual*rho_a + 2*lam * I
        hess += 8 * np.outer(rho_a, rho_a) - 4*residual*rho + 2*lam * np.eye(n)

    return grad, hess



def get_lin (corrs, a, beams, lam):

    n = len(a)
    m = len(beams)

    R = np.empty(shape = (m, n))

    for p in range(m):
        for k in range(n):
            R[p, k] = beams[p, k, k]
    

    RR = R.T @ R

    return np.linalg.inv(RR + lam*np.eye(n)) @ R.T @ corrs






if __name__ == "__main__":

    n = 50
    i = 1
    j = 2
    center = [60, 30] * u.deg

    width = [2e-1, 2e-1] * u.deg

    ones = np.ones(shape = (n, n))


    # make a grid of explored pixels
    phi = center[0] + np.linspace(-width[0]/2, width[0]/2, n)
    theta = center[1] + np.linspace(-width[1]/2, width[1]/2, n)

    phi, theta = np.meshgrid(phi, theta)
    phi_r = phi.ravel()
    theta_r = theta.ravel()


    grid = GWSource(theta = theta, 
                      phi = phi,
                      frequency = 1e-8 * u.Hz * ones,
                      strain = ones)

    grid_r = GWSource(theta = theta_r, 
                      phi = phi_r,
                      frequency = 1e-8 * u.Hz * np.ones(shape = (n*n, n*n)),
                      strain = np.ones(shape = (n*n, n*n)))

    center = GWSource(theta = center[1], 
                      phi = center[0],
                      frequency = 1e-8 * u.Hz,
                      strain = 1)




    pulsars = Pulsar.load_collection("pulsars")
    # psr1 = Pulsar(0*u.deg, 0*u.deg, 1*u.kpc)
    # psr2 = Pulsar(90*u.deg, 0*u.deg, 1*u.kpc)
    # pulsars = [psr1, psr2]

    pta = PTACorrelationDetector(pulsars)
    n_pul = len(pulsars) 
    n_pul = 10

    # Find apropriate phases to poit the antenna

    delta_phi = np.empty(n_pul*(n_pul+1)//2) * u.rad 

    pp = np.linspace(0, 2*np.pi, 100) * u.rad
    c = np.empty_like(pp)

    for i in range(n_pul):
        for j in range(i, n_pul):

            c = corr(pulsars[i],
                       pulsars[j],
                       pp/center.frequency)


            idx = np.argmax(c)
            arg = get_arg(i, j, n_pul)

            delta_phi[arg] = pp[idx]

    # prepare beam patterns for all pulsar pairs
    beams = np.empty(shape = (n_pul*(n_pul+1)//2, n*n, n*n))
    corrs = np.empty(shape = (n_pul*(n_pul+1)//2))
    


    for i in range(n_pul):
        for j in range(i, n_pul):

            arg = get_arg(i, j, n_pul)
            print(f"{arg}/{n_pul*(n_pul+1)//2}")

            beams[arg] = pta.pair_beam_pattern(i, j, 
                                               grid_r, 
                                               delta_phi[arg])
            corrs[arg] = corr(pulsars[i],
                              pulsars[j],
                              delta_phi[arg]/center.frequency)


    a = np.ones(shape = (n, n), dtype = np.float64)
    a = a.ravel()

    a = get_lin(corrs, a, beams, 1e-2)

    plt.figure()
    plt.imshow(a.reshape((n, n)),
               origin = "lower",
               cmap = "hot",
               )

    #plt.plot(n/2, n/2, "*")
    plt.colorbar()

#    for i in range(10):
#        print(i)
#
#        grad, hess = grad_hess(corrs, a, beams, 1)
#        a = a - np.linalg.inv(hess) @ grad
#
#        plt.figure()
#        plt.imshow(a.reshape((n, n)),
#                   origin = "lower",
#                   cmap = "hot")
#        plt.plot(n/2, n/2, "*")
#        plt.colorbar()




    save_image("123.pdf")







