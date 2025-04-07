# GW-Interferometry: Gravitational Wave Pulsar Timing Array Analysis

## Overview

This repository provides a Python-based framework for simulating and analyzing gravitational wave (GW) signals using pulsar timing arrays (PTAs). It includes tools to:

- Simulate pulsar timing data affected by gravitational waves.
- Model gravitational wave sources and their effects on pulsar signals.
- Detect and reconstruct gravitational wave signals via correlation analysis.
- Generate sky maps of gravitational wave sources using imaging techniques.


## Features

- **Gravitational Wave Simulation**:
  - Generate gravitational wave sources (`GWSource`) with configurable parameters such as frequency, strain, and sky position.
  - Simulate the effect of gravitational waves on pulsar timing data.

- **Pulsar Modeling**:
  - Create pulsar objects (`Pulsar`) with realistic sky positions and distances.
  - Generate observation schedules and compute GW-induced redshifts for pulsars.

- **Detection and Imaging**:
  - Perform correlation-based detection of gravitational wave signals in PTA data.
  - Reconstruct sky images of gravitational wave sources using Tikhonov regularization.

- **Visualization**:
  - Includes tools for creating high-quality plots of results using `matplotlib`.


## Repository Structure

- **`generate.py`**
  Contains classes for simulating gravitational wave sources and pulsars. This module includes methods to generate observation times, compute GW-induced redshifts, and save/load pulsar data from HDF5 files.

- **`detect.py`**
  Implements the detection framework for PTAs. It includes functionality to compute pairwise beam patterns, determine optimal phase differences for each pulsar pair, and solve the inverse problem to reconstruct GW sky maps.

- **`myplot.py`**
  Provides helper functions for plotting and saving figures in PDF format with LaTeX support.

- **`requirements.txt`**
  Lists all the necessary Python dependencies to run the simulator and detector.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/gw-interferometry.git
    cd gw-interferometry
    ```

2. **Create and activate a Python virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Simulation of PTA Data

To generate simulated pulsar data and add the redshift induced by a gravitational wave, run the `generate.py` script:

```bash
python generate.py
```

This will create a collection of pulsars (saved as individual HDF5 files in the `pulsars` directory) with simulated observation times and corresponding redshift data.

### GW Detection and Imaging

The main detection and imaging routines are implemented in `detect.py`. A typical workflow is:

1. **Load Pulsars:**
Load a collection of pulsars from the `pulsars` directory:
```python3
from generate import Pulsar
pulsars = Pulsar.load_collection("pulsars")
```

2. **Initialize the Detector:**
Create a `PTACorrelationDetector` instance with the pulsars:
```python3
from detect import PTACorrelationDetector
pta = PTACorrelationDetector(pulsars)
```

3. **Point the Detector:**
Choose a GW source (pointed at some center position) and find the optimal phase differences:
```python3
from generate import GWSource
import astropy.units as u
center = GWSource(theta=30*u.deg, phi=60*u.deg, frequency=1e-8*u.Hz, strain=1)
pta.point_detector(center)
```

4. **Image Reconstruction:**
Create a grid of sky positions and reconstruct the sky image:
```python3
n = 100
# Define the field-of-view center and width
center_pos = [60, 30] * u.deg
width = [12, 12] * u.arcmin
# Create the grid (meshgrid for phi and theta)
import numpy as np
phi = center_pos[0] + np.linspace(-width[0]/2, width[0]/2, n)
theta = center_pos[1] + np.linspace(-width[1]/2, width[1]/2, n)
phi, theta = np.meshgrid(phi, theta)
grid = GWSource(theta=theta, phi=phi, frequency=1e-8*u.Hz, strain=1)

# Reconstruct and plot the image
img = pta.image_point(grid, lam=1e0)
```

5. **Plotting:**
The provided plotting routines in `myplot.py` help generate and save publication-quality figures.

## Contributing

Contributions are welcome! Please open issues or submit pull requests with improvements, bug fixes, or additional features.

## License

This repository is licensed under the MIT License. See the `LICENSE` file for details.



