from scipy.stats import skewnorm
from collections import namedtuple
import numpy as np
import healpy as hp
import camb
import csv
import os
from astropy.io import fits

def read_map(file_path):
    """
    Reads a Healpy map from a FITS file and flattens the data.
    """
    
    with fits.open(file_path) as hdul:
        hdul.info()
        if len(hdul) > 1 and hasattr(hdul[1], 'columns'):
            print(hdul[1].columns)
        return np.concatenate(hdul[1].data['T'])