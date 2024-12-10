from dataclasses import dataclass

import numpy as np
import unyt as u


@dataclass(kw_only=True, slots=True)
class Header:
    scale: float
    redshift: float
    h: float
    H: u.unyt_quantity
    box_size: u.unyt_array
    num_part: np.ndarray
    Omega_cdm: float
    Omega_b: float
    Omega_m: float
    Omega_Lambda: float
