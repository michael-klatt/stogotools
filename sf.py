#! /usr/bin/env python
# Author: Michael A. Klatt
# Version 19.03

import numpy as np
import math
import sys
import argparse

tqdm_available = True
try:
    from tqdm import tqdm
except ImportError:
    tqdm_available = False

# Parse parameters
descrpt = "sy.py estimates the structure factor for a 2D or 3D point pattern in a cubic simulation box with periodic boundary conditions by computing the scattering intensity. Author: Michael A. Klatt, Contact: software@mklatt.org"
parser = argparse.ArgumentParser(description=descrpt)
parser.add_argument('infile',
                    help='Filename of CSV file containing the coordinates')
parser.add_argument('-s', '--sctint', action="store_true",
                    help='Output unbinned scattering intensity')
parser.add_argument('--log', action="store_true",
                    help='Logarithmic binning')
parser.add_argument('-L', '--system-size', metavar='float', type=float,
                    help='Linear system size (default: assumes unit number density)')
parser.add_argument('-n', '--nbins', metavar='int', type=int, default=40,
                    help='Number of bins (default: 40)')
parser.add_argument('--min-k', metavar='float', type=float, default=0.0,
                    help='Minimum wavenumber (default: 0.0)')
parser.add_argument('--max-k', metavar='float', type=float, default=16.5,
                    help='Maximum wavenumber (default: 16.5)')
parser.add_argument('-c', '--skip-columns', metavar='int', type=int, default=0,
                    help='Skip this number of columns (default: 0)')
args = parser.parse_args()

coords_file  = args.infile
L            = args.system_size
N_k_bins     = args.nbins
min_k        = args.min_k
max_k        = args.max_k
skip_columns = args.skip_columns
log_binning  = args.log

# Read coordinates
print("Reading in", coords_file, "...")
try:
    points = np.loadtxt(coords_file)[:,skip_columns:]
except OSError:
    print("ERROR: input file", coords_file, "not found.")
    sys.exit(-1)

dimension = np.shape(points)[1] 
N_points = np.shape(points)[0]
    
print('Preparing structure factor ...')
if dimension == 2:
    if L == None:
        L = math.sqrt(N_points)
        
    max_N = int((max_k-min_k)/(2*math.pi/L))+1

    N_k = (2*max_N)**2
    
    k  = np.zeros((N_k))
    kx = np.zeros((N_k))
    ky = np.zeros((N_k))
    nxny = np.zeros((N_k,2))
    
    delta_k_0 = 2*math.pi/L
    delta_k_1 = 2*math.pi/L
    
    count = 0
    tmp = np.zeros((2,1))
    for nx in range(1,max_N):
        ny = 0
        tmp[0] = nx*delta_k_0
        tmp[1] = ny*delta_k_1
    
        tmp_k = np.linalg.norm(tmp)

        if tmp_k < max_k and tmp_k > min_k:
            kx[count] = tmp[0]
            ky[count] = tmp[1]
            k[count]  = tmp_k

            nxny[count,0] = nx
            nxny[count,1] = ny

            count += 1
    for nx in range(-max_N,max_N):
        for ny in range(1,max_N):
    
            tmp[0] = nx*delta_k_0
            tmp[1] = ny*delta_k_1
        
            tmp_k = np.linalg.norm(tmp)

            if tmp_k < max_k and tmp_k > min_k:
                kx[count] = tmp[0]
                ky[count] = tmp[1]
                k[count]  = tmp_k

                nxny[count,0] = nx
                nxny[count,1] = ny

                count += 1
    
    kx = kx[:count]
    ky = ky[:count]
    k  = k[:count]
    nxny = nxny[:count,:]
       
    print('Determining structure factor ...')
    S = np.zeros((count), dtype=complex) # structure factor
    if tqdm_available:
        for sj in tqdm(range(N_points)):
            S += np.exp(-1j * (kx*points[sj,0]+ky*points[sj,1]) )
    else:
        for sj in range(N_points):
            S += np.exp(-1j * (kx*points[sj,0]+ky*points[sj,1]) )
    
    # The final (unbinned) structure factor for each absolute value of k
    unbinned = np.zeros((count,4))
    unbinned[:,0]  = k
    unbinned[:,1]  = np.square(np.absolute(S))/ N_points
    unbinned[:,2:] = nxny

elif dimension == 3:
    if L == None:
        L = N_points**(1./3)
        
    max_N = int((max_k-min_k)/(2*math.pi/L))+1

    N_k = (2*max_N)**3
    
    k  = np.zeros((N_k))
    kx = np.zeros((N_k))
    ky = np.zeros((N_k))
    kz = np.zeros((N_k))
    nxnynz = np.zeros((N_k,3))
    
    delta_k_0 = 2*math.pi/L
    delta_k_1 = 2*math.pi/L
    delta_k_2 = 2*math.pi/L
    
    count = 0
    tmp = np.zeros((3,1))
    for nx in range(-max_N,max_N):
        for ny in range(-max_N,max_N):
            nz = 0 

            if ny > -nx or (ny == -nx and nx > 0):
                tmp[0] = nx*delta_k_0
                tmp[1] = ny*delta_k_1
                tmp[2] = nz*delta_k_2
            
                tmp_k = np.linalg.norm(tmp)

                if tmp_k < max_k and tmp_k > min_k:
                    kx[count] = tmp[0]
                    ky[count] = tmp[1]
                    kz[count] = tmp[2]
                    k[count]  = tmp_k

                    nxnynz[count,0] = nx
                    nxnynz[count,1] = ny
                    nxnynz[count,2] = nz

                    count += 1
    for nx in range(-max_N,max_N):
        for ny in range(-max_N,max_N):
            for nz in range(1,max_N):
    
                tmp[0] = nx*delta_k_0
                tmp[1] = ny*delta_k_1
                tmp[2] = nz*delta_k_2
            
                tmp_k = np.linalg.norm(tmp)

                if tmp_k < max_k and tmp_k > min_k:
                    kx[count] = tmp[0]
                    ky[count] = tmp[1]
                    kz[count] = tmp[2]
                    k[count]  = tmp_k

                    nxnynz[count,0] = nx
                    nxnynz[count,1] = ny
                    nxnynz[count,2] = nz

                    count += 1
    
    kx = kx[:count]
    ky = ky[:count]
    kz = kz[:count]
    k  = k[:count]
    nxnynz = nxnynz[:count,:]
       
    print('Determining structure factor ...')
    S = np.zeros((count), dtype=complex) # structure factor
    if tqdm_available:
        for sj in tqdm(range(N_points)):
            S += np.exp(-1j * (kx*points[sj,0]+ky*points[sj,1]+kz*points[sj,2]) )
    else:
        for sj in range(N_points):
            S += np.exp(-1j * (kx*points[sj,0]+ky*points[sj,1]+kz*points[sj,2]) )
    
    # The final (unbinned) structure factor for each absolute value of k
    unbinned = np.zeros((count,5))
    unbinned[:,0]  = k
    unbinned[:,1]  = np.square(np.absolute(S))/ N_points
    unbinned[:,2:] = nxnynz
else:
    print("ERROR: the script currently works only for 2D and 3D,")
    print("       but the number of columns is", dimension)
    print("       Are there columns that should be skipped?")
    sys.exit(-1)

# Output scattering intensity? 
if args.sctint:
    if dimension == 2:
        np.savetxt(coords_file[:-4] + ".sctint", unbinned, fmt='%.10f %.10f %i %i')
    elif dimension == 3:
        np.savetxt(coords_file[:-4] + ".sctint", unbinned, fmt='%.10f %.10f %i %i %i')

# Binning to estimate structure factor
if log_binning:
    if min_k == 0.0:
        print("Logarithmic binning automatically uses minimum wavenumber 0.01 instead of 0.")
        min_k = 0.01
    b = 10.0
    bins = np.logspace(np.log(min_k)/np.log(b),np.log(max_k)/np.log(b),num=N_k_bins+1,base=b)
else:
    bins = np.linspace(min_k,max_k+1,N_k_bins+1)
binwidth = bins[1:]-bins[:-1]

idx = np.digitize(unbinned[:,0], bins)

# output should contain
# k(mean of bin)   <S>   err(S)   N_entries_in_this_bin
output = np.zeros((np.size(bins)-1,4))
output[:,0] = (bins[1:]+bins[:-1])*0.5
for i,bin_j in enumerate(idx):
    # i is row in unbinned, bin_j is bin to which it belongs
    output[bin_j-1,1] += unbinned[i,1]
    output[bin_j-1,2] += (unbinned[i,1])**2
    output[bin_j-1,3] += 1

# only evaluate bins with more than one k-value
idx = np.where(output[:,3]>1)[0]

output[idx,2] = np.sqrt(np.fabs(output[idx,2]/(output[idx,3]-1)/output[idx,3] - np.power(output[idx,1],2)/output[idx,3]/(output[idx,3]-1)/output[idx,3]))
output[idx,1] /= output[idx,3]

# for all others assign nan
idx = np.where(output[:,3]<2)[0]
output[idx,1:3] = np.nan

# Output structure factor
np.savetxt(coords_file[:-4] + "-Sk.dat", output, fmt="%.10f %.10f %.10f %i")

