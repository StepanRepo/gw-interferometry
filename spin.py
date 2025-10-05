#! /bin/python


import numpy as np
from scipy.special import loggamma as lgm, logsumexp
from alp_integral import logI2

#import matplotlib.pyplot as plt
#from matplotlib.transforms import Affine2D

#import spherical
#import quaternionic


def plot_sphere(theta, phi, img, title = None, colorbar = True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='mollweide')
    cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])

    x = phi - np.pi                         # Wrap to [-π, π]
    y = np.pi/2 - theta                     # Latitude ([-π/2, π/2])

    pcm = ax.pcolormesh(x, y, img, 
                        cmap = "hot",
                        )

    if colorbar:
        fig.colorbar(pcm, cax=cax, orientation='horizontal')

    if title is not None:
        cax.set_title(title)

    return fig, ax

def E1p(l, m, l1):
    m1 = -m
    logI, sign_I = logI2(l1-2, m1+2, l, m)

    if sign_I == 0.0:
        return -np.inf, 0.0

    log_coef = np.log(abs((l1 - m1**2) / (4*m1) / (m1+1)))
    sign_coef = np.sign((l1 - m1**2) / (4*m1) / (m1+1))
    return logI + log_coef, sign_I * sign_coef

def E2p(l, m, l1):
    m1 = -m
    logI, sign_I = logI2(l1-2, m1, l, m)

    if sign_I == 0.0:
        return -np.inf, 0.0

    log_coef = np.log(abs((l1 - m1**2) * (l1 + m1 - 1) * (l1 + m1)/ 2.0 / (m1**2 - 1)))
    sign_coef = np.sign((l1 - m1**2) * (l1 + m1 - 1) * (l1 + m1)/ 2.0 / (m1**2 - 1))
    return logI + log_coef, sign_I * sign_coef

def E3p(l, m, l1):
    m1 = -m
    logI, sign_I = logI2(l1-2, m1-2, l, m)

    if sign_I == 0.0:
        return -np.inf, 0.0

    term1 = (l1 + m1) * (l1 + m1 - 1) * (l1 + m1 - 2) * (l1 + m1 - 3)
    log_coef = np.log(abs(term1 * (l1 - m1**2) / (4*m1) / (m1-1)))
    sign_coef = np.sign(term1 * (l1 - m1**2) / (4*m1) / (m1-1))
    return logI + log_coef, sign_I * sign_coef

def E4p(l, m, l1):
    m1 = -m
    logI, sign_I = logI2(l1, m1, l, m)

    if sign_I == 0.0:
        return -np.inf, 0.0

    log_coef = np.log(abs(.5 * l1 * (l1 - 1)))
    sign_coef = np.sign(.5 * l1 * (l1 - 1))
    return logI + log_coef, sign_I * sign_coef

def E5p(l, m, l1):
    m1 = -m
    logI, sign_I = logI2(l1-2, m1+2, l, m)

    if sign_I == 0.0:
        return -np.inf, 0.0

    log_coef = np.log(abs(-(l1 + m1) / (4*m1) / (m1 + 1)))
    sign_coef = np.sign(-(l1 + m1) / (4*m1) / (m1 + 1))
    return logI + log_coef, sign_I * sign_coef

def E6p(l, m, l1):
    m1 = -m
    logI, sign_I = logI2(l1-2, m1, l, m)

    if sign_I == 0.0:
        return -np.inf, 0.0

    log_coef = np.log(abs(-(l1+m1) * (l1-1) * (l1 + m1 - 1) / 2.0 / (m1**2 - 1)))
    sign_coef = np.sign(-(l1+m1) * (l1-1) * (l1 + m1 - 1) / 2.0 / (m1**2 - 1))
    return logI + log_coef, sign_I * sign_coef

def E7p(l, m, l1):
    m1 = -m
    logI, sign_I = logI2(l1-2, m1-2, l, m)

    if sign_I == 0.0:
        return -np.inf, 0.0

    term1 = (l1 + m1 - 1) * (l1 + m1 - 2) * (l1 + m1 - 3) 
    log_coef = np.log(abs(-term1 * (l1+m1) * (l1-m1) / (4*m1) /(m1-1)))
    sign_coef = np.sign(-term1 * (l1+m1) * (l1-m1) / (4*m1) /(m1-1))
    return logI + log_coef, sign_I * sign_coef

def E1m(l, m, l1):
    m1 = -m
    logI, sign_I = logI2(l1-1, m1+2, l, m)

    if sign_I == 0.0:
        return -np.inf, 0.0

    log_coef = np.log(abs((l1 - 1) / (4 * (m1 + 1))))
    sign_coef = np.sign((l1 - 1) / (4 * (m1 + 1)))
    return logI + log_coef, sign_I * sign_coef

def E2m(l, m, l1):
    m1 = -m
    logI, sign_I = logI2(l1-1, m1, l, m)

    if sign_I == 0.0:
        return -np.inf, 0.0

    log_coef = np.log(abs((l1 - 1)/2 * l1 * m1 * (l1 + m1) / (m1**2 - 1)))
    sign_coef = np.sign((l1 - 1)/2 * l1 * m1 * (l1 + m1) / (m1**2 - 1))
    return logI + log_coef, sign_I * sign_coef

def E3m(l, m, l1):
    m1 = -m
    logI, sign_I = logI2(l1-1, m1-2, l, m)

    if sign_I == 0.0:
        return -np.inf, 0.0

    term1 = (l1 + m1) * (l1 + m1 - 1) * (l1 + m1 - 2)
    term2 = (l1 - m1 + 1)
    log_coef = np.log(abs((l1 - 1) * term1 * term2 / (4 * (m1 - 1))))
    sign_coef = np.sign((l1 - 1) * term1 * term2 / (4 * (m1 - 1)))
    return logI + log_coef, sign_I * sign_coef

def E4m(l, m, l1):
    m1 = -m
    logI, sign_I = logI2(l1-3, m1+2, l, m)

    if sign_I == 0.0:
        return -np.inf, 0.0

    log_coef = np.log(abs(-(l1 + m1) / (4 * (m1 + 1))))
    sign_coef = np.sign(-(l1 + m1) / (4 * (m1 + 1)))

    return logI + log_coef, sign_I * sign_coef

def E5m(l, m, l1):
    m1 = -m
    logI, sign_I = logI2(l1-3, m1, l, m)

    if sign_I == 0.0:
        return -np.inf, 0.0

    log_coef = np.log(abs(-(l1+m1)/2 * m1 * (l1+m1-2) * (l1+m1-1) / (m1**2 - 1)))
    sign_coef = np.sign(-(l1+m1)/2 * m1 * (l1+m1-2) * (l1+m1-1) / (m1**2 - 1))
    return logI + log_coef, sign_I * sign_coef

def E6m(l, m, l1):
    m1 = -m
    logI, sign_I = logI2(l1-3, m1-2, l, m)

    if sign_I == 0.0:
        return -np.inf, 0.0

    term1 = (l1 + m1 - 1) * (l1 + m1 - 2) * (l1 + m1 - 3) * (l1 + m1 - 4)
    log_coef = np.log(abs(-(l1+m1)/4 * term1 / (m1 - 1)))
    sign_coef = np.sign(-(l1+m1)/4 * term1  / (m1 - 1))

    return logI + log_coef, sign_I * sign_coef


def intY2Y(l, m, l1):
    m1 = -m

    logN_l1 = .5 * (lgm(l1-1) - lgm(l1+3))

    logK_l1 = .5 * np.log((2*l1 + 1)/(4*np.pi)) + \
            .5 * (lgm(l1 - m1 + 1) - lgm(l1 + m1 +1))

    logK_l = .5 * np.log((2*l + 1)/(4*np.pi)) + \
            .5 * (lgm(l - m + 1) - lgm(l + m +1))

    log_C = np.log(2) + logN_l1 + logK_l1 + logK_l
    sign_C = -1.0
    #coef2 = (-1)**m * np.exp(lgm(l+m+1) - lgm(l-m+1)) / (2*np.pi)

    E_terms = np.array([
        E1p(l, m, l1), 
        E2p(l, m, l1), 
        E3p(l, m, l1), 
        E4p(l, m, l1),
        E5p(l, m, l1), 
        E6p(l, m, l1), 
        E7p(l, m, l1),
        E1m(l, m, l1), 
        E2m(l, m, l1), 
        E3m(l, m, l1),
        E4m(l, m, l1), 
        E5m(l, m, l1), 
        E6m(l, m, l1)
    ])


    print(E_terms)
    E_terms = E_terms[E_terms[:, 1] != 0.0]
    logE = E_terms[:, 0]
    signs = E_terms[:, 1]
    print(np.exp(logE)*signs)


    if len(logE) == 0:
        return 0.0


    summ, sign = logsumexp(logE, b=signs, return_sign=True)
    summ = summ + log_C 
    sign = sign * sign_C

    return sign * np.exp(summ)


if __name__ == "__main__":

    l1 = 5
    l = 5
    m = 2
    analytica = intY2Y(l, m, l1)

    print(f"Re analytica = {analytica}")





#    fig, ax = plot_sphere(theta, phi, Y2[:, :, idx].real, title = f"${{}}_2 Y_{{{l},{m}}}$")
#    fig, ax = plot_sphere(theta, phi,  Y[:, :, idx].real, title = f"$Y_{{{l},{m}}}$")
#
#
#    fig, ax = plot_sphere(theta, phi, (Y2[:, :, idx] * Y[:, :, idx]).real, 
#                          title = f"${{}}_2 Y_{{{l},{m}}} \cdot Y_{{{l},{m}}}$")
#    
#
#    
#
#    save_image("123.pdf")
