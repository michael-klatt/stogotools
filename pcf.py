#!/usr/bin/env python
# Author: Michael A. Klatt
# Version 19.09

import numpy as np
import pandas as pd
import math
import sys
import argparse

from scipy import stats
from scipy.optimize import minimize

try:
    from tqdm import tqdm
    tqdm_available = True
    # Print progress via tqdm(range(10))
except ImportError:
    tqdm_available = False

def pairwise_distances(x,N_pts):
    """Compute pwd"""
    x = np.mod(x,1) # Choose representative in unit cube
    diff_matrix = np.abs(x[:, :, None] - x[:, :, None].T) # based on stackoverflow.com/questions/28687321/computing-euclidean-distance-for-numpy-in-python
    diff_complm = 1 - diff_matrix
    diff = np.minimum.reduce([diff_matrix, diff_complm])
    D = np.sqrt( (diff**2).sum(1) )
    return D[np.triu_indices(N_pts,k=1)]

def coords2pcf(coords_file, L, N_r_bins):
    """Load coordinates & Return pcf"""
    # Load coordinates
    x = np.loadtxt(coords_file)
    N_pts = np.shape(x)[0]
    dim = np.shape(x)[1]

    # Convert to unit cube as simulation box
    x /= L

    # Compute pairwise distances
    r = pairwise_distances(x,N_pts)

    # Histogram estimate of pcf
    max_r = 0.5

    bins = np.linspace(0,max_r+1,N_r_bins)
    binwidth = np.mean((bins[1:]-bins[:-1]))
    
    min_w0 = 0
    max_w0 = max_r
    number_of_bins_w0 = N_r_bins
    
    bin_width_w0 = (max_w0 - min_w0)/number_of_bins_w0
    o_xedges = np.linspace(min_w0,max_w0,number_of_bins_w0,endpoint=True)
    
    number_of_hits, xedges = np.histogram(r, bins=o_xedges, density=False)
    error = np.sqrt(number_of_hits*(1-number_of_hits*1.0/np.size(r)))
    
    number_of_hits *= 2
    error *= 2

    g2 = number_of_hits.astype(float)/N_pts
    error /= N_pts

    v_shell = v_ball(dim)*(xedges[1:]**2-xedges[:-1]**2)
    
    g2 /= v_shell*N_pts
    error /= v_shell*N_pts
    
    lx = len(xedges) - 1
    histogram_table = np.zeros( (lx, 3) )
    # Scale back to original system size
    histogram_table[:,0] = L*np.transpose(xedges[:-1]+bin_width_w0/2.0)
    histogram_table[:,1] = np.transpose(g2)
    histogram_table[:,2] = error

    return histogram_table

def v_ball(dimension):
    """Volume of unit ball"""
    return math.pi**(dimension*0.5)/math.gamma(dimension*0.5+1)

def main(args):
    """Main function"""
    pcf = coords2pcf(args.infile, args.L, args.N)
    np.savetxt(args.infile[:-4] + '-pcf.dat', pcf, fmt='%.15f')

def parse():
    """Parse parameters from command line"""
    d = "This script computes the pairwise distances (pwd) for a point pattern in a cubic simulation box with periodic boundary conditions."
    parser = argparse.ArgumentParser(description=d)
    parser.add_argument('infile',
                        help='Filename of input coordinates')
    parser.add_argument('-L', default=1.0, type=float, metavar='float',
                        help='Linear system size (default 1.0)')
    parser.add_argument('-N', default=100, type=int, metavar='int',
                        help='Number of bins (default 100)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    main(args)

