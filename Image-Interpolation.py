#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract subimages for different polarizer orientations from a raw image file.
Four subimages are extracted based on the pixel orientations of 0, 45, 90, and 135 degrees.
The subimages are then interpolated to fill missing (NaN) values.
The degree of linear polarization (DoLP) and angle of linear polarization (AoLP) are computed.
The results are visualized and saved as a PDF file.

Before running the code:
(a) Modify the variable 'datadir' below to point to the folder where the raw image is located.
(b) Modify the variable 'ipfile' to point to the raw image file.

Code output:
If the code runs without any errors, it will produce a PDF file containing visualizations
of the original image, interpolated subimages, DoLP, and AoLP.

"""

import os
import sys
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.visualization import (
    ManualInterval, LinearStretch, ImageNormalize)

# Image dimensions
img_height, img_width = 2048, 2448
NPIX = img_height * img_width  # Total number of pixels

# File and folder paths
datadir = '/Users/ahadibrahimi/Downloads/Image-Interpolation'
ipfile = 'asun20ms.raw'
myfile = os.path.join(datadir, ipfile)
opfnam = ipfile + '.pdf'
opfile = os.path.join(datadir, opfnam)

######################################################################
# Read the raw image data into a 2D numpy image array

# Read binary data from raw file into 8-bit unsigned integer numpy array
binary_array = np.fromfile(myfile, dtype='uint8')

num_bytes = len(binary_array)

if (num_bytes == NPIX):          # Mono8 format

    bpp = 8   # Bits/pixel
    print('Decoding 8-bit image:', ipfile)
    img_arr_1d = binary_array

elif (num_bytes == 2*NPIX):      # Mono12 format

    bpp = 12   # Bits/pixel
    print('Decoding 12-bit image:', ipfile)

    # These are the even bytes, i.e. the 0th, 2nd, 4th, 6th, 8th, 10th, ...
    evn_arr = binary_array[0::2].copy()

    # These are the odd bytes, i.e. the 1st, 3rd, 5th, 7th, 9th, 11th, ...
    odd_arr = binary_array[1::2].copy()

    # Recast these arrays as 16-bit integer arrays because an 8-bit integer
    # array can store from 0 to 255 only whereas we need to store bigger
    # numbers when working with 12-bit integers.
    evn_arr = evn_arr.astype('int16')
    odd_arr = odd_arr.astype('int16')

    # Shift the odd bytes 8 binary digits and then add to the even byte to
    # get the 12-bit pixel value.
    pix_val = (odd_arr << 8) + evn_arr
    img_arr_1d = pix_val

else:

    print('Unrecognized image format')
    sys.exit(0)

# Reshape 1D array img_arr_1d to 2D image array
img_array = img_arr_1d.reshape(img_height, img_width)

# Done reading raw image into a 2D array.
######################################################################
# Extract the 4 subimages, one for each polarizer orientation

orient_090 = img_array[::2, ::2].astype('float64')
orient_045 = img_array[::2,  1::2].astype('float64')
orient_135 = img_array[1::2, ::2].astype('float64')
orient_000 = img_array[1::2, 1::2].astype('float64')

# Done extracting subimages
######################################################################

# Populating the subarrays into the original size matrix

# Create empty original-sized matrices filled with NaNs
original_size = img_height, img_width
orient_090_resized = np.full(original_size, np.nan)
orient_045_resized = np.full(original_size, np.nan)
orient_135_resized = np.full(original_size, np.nan)
orient_000_resized = np.full(original_size, np.nan)

# Define the positions to populate the subarray into the resized matrix
rows_090_idx = slice(0, orient_090_resized.shape[0], 2)
cols_090_idx = slice(0, orient_090_resized.shape[1], 2)
rows_045_idx = slice(0, orient_045_resized.shape[0], 2)
cols_045_idx = slice(1, orient_045_resized.shape[1], 2)
rows_135_idx = slice(1, orient_135_resized.shape[0], 2)
cols_135_idx = slice(0, orient_135_resized.shape[1], 2)
rows_000_idx = slice(1, orient_000_resized.shape[0], 2)
cols_000_idx = slice(1, orient_000_resized.shape[1], 2)

# Populate the subarrays into the resized matrices
orient_090_resized[rows_090_idx, cols_090_idx] = orient_090
orient_045_resized[rows_045_idx, cols_045_idx] = orient_045
orient_135_resized[rows_135_idx, cols_135_idx] = orient_135
orient_000_resized[rows_000_idx, cols_000_idx] = orient_000

#########################################################################
# Interpolating the empty cells using the populated cells from the subarrays in the original size


def interpolate_matrix(arr):
    """
    Perform linear interpolation to fill in missing (NaN) values along each row and column of a 2D array.
    Parameters:
        arr (numpy.ndarray): 2D array containing missing values as NaN.
    Returns:
        numpy.ndarray: Modified array with missing values filled in.
    """
    arr = np.array(arr)  # Convert input list to a NumPy array

    # Interpolate in rows
    rows, cols = arr.shape
    for i in range(rows):
        mask = np.isnan(arr[i])
        indices = np.where(~mask)[0]
        if len(indices) > 1:
            arr[i, mask] = np.interp(
                np.where(mask)[0], indices, arr[i, indices])

    # Interpolate in columns
    for i in range(cols):
        mask = np.isnan(arr[:, i])
        indices = np.where(~mask)[0]
        if len(indices) > 1:
            arr[mask, i] = np.interp(
                np.where(mask)[0], indices, arr[indices, i])

    return arr


# Interpolate the matrices
orient_090_interp = interpolate_matrix(orient_090_resized)
orient_045_interp = interpolate_matrix(orient_045_resized)
orient_135_interp = interpolate_matrix(orient_135_resized)
orient_000_interp = interpolate_matrix(orient_000_resized)

print("Interpolation completed.")
#########################################################################

# DoLP and AoLP computations
S0 = (orient_000_interp + orient_045_interp +
      orient_090_interp + orient_135_interp)/2
S1 = orient_000_interp - orient_090_interp
S2 = orient_045_interp - orient_135_interp
DoLP = 100.0*(np.sqrt((S1*S1) + (S2*S2)))/S0
AoLP = np.rad2deg(0.5 * np.arctan2(S2, S1))

######################################################################
# Visualization of results

# use copy so that we do not mutate the global colormap instance
mycmap = copy(plt.cm.gray)

# Location and size of the main image [xmin, ymin, dx, dy]
main_axis = [0.05, 0.05, 0.75, 0.85]

###############
# Arrays needed to create the half-circular colorbar for AoLP maps

# Radial grid
rmin, rmax, rpts = 0.7, 1.0, 100
radii = np.linspace(rmin, rmax, rpts)

# theta values on the right side of the color circle
thpts = 500
azimuthsR = np.linspace(-90, 91, thpts)
valuesR = azimuthsR * np.ones((rpts, thpts))
###############

print('Creating plots')

with PdfPages(opfile) as pdf:

    # Plot the raw image
    fig = plt.figure(figsize=(10, 8), dpi=200)
    ax0 = fig.add_axes(main_axis)
    cbar_label = 'Counts (ADU)'
    ax0.set_title('Raw image from sensor')
    im = ax0.imshow(img_array, origin='upper', cmap=mycmap)

    ax1 = fig.add_axes([0.85, 0.10, 0.04, 0.75])
    cbar = fig.colorbar(im, cax=ax1)
    cbar.set_label(cbar_label)
    pdf.savefig(fig)
    plt.close()

    # Plot the interpolated subimages
    for interp_orient, title in zip([orient_000_interp, orient_045_interp, orient_090_interp, orient_135_interp],
                                    ['$0^\circ$', '$45^\circ$', '$90^\circ$', '$135^\circ$']):
        fig = plt.figure(figsize=(10, 8), dpi=200)
        ax0 = fig.add_axes(main_axis)
        cbar_label = 'Counts (ADU)'
        ax0.set_title(
            f'Interpolated Subimage with pixels oriented {title}, measured CCW from horizontal')
        im = ax0.imshow(interp_orient, origin='upper', cmap=mycmap)

        ax1 = fig.add_axes([0.85, 0.10, 0.04, 0.75])
        cbar = fig.colorbar(im, cax=ax1)
        cbar.set_label(cbar_label)
        pdf.savefig(fig)
        plt.close()

    # Visualization of DoLP
    mycmap = copy(plt.cm.viridis)
    mycmap.set_bad('w', 1.0)

    fig = plt.figure(figsize=(10, 8), dpi=200)
    dolpmin, dolpmax = np.amin(DoLP), np.amax(DoLP)
    mynorm = ImageNormalize(DoLP, interval=ManualInterval(dolpmin, dolpmax),
                            stretch=LinearStretch())
    ax0 = fig.add_axes(main_axis)
    ax0.set_title(
        'Degree of Linear Polarization (white pixels were nonlinear and excluded)')
    im = ax0.imshow(DoLP, origin='upper', norm=mynorm, cmap=mycmap)

    ax1 = fig.add_axes([0.85, 0.10, 0.04, 0.75])
    cbar = fig.colorbar(im, cax=ax1)
    cbar.set_label('Degree of linear polarization (percent)')
    pdf.savefig(fig)
    plt.close()

    # Visualization of AoLP
    mycmap = copy(plt.cm.hsv)
    mycmap.set_bad('w', 1.0)
    fig = plt.figure(figsize=(10, 8), dpi=200)
    ax0 = fig.add_axes(main_axis)
    ax0.set_title(
        'Angle of Linear Polarization (white pixels were nonlinear and excluded)')
    im = ax0.imshow(AoLP, origin='upper', cmap=mycmap)

    ax1 = fig.add_axes([0.72, 0.45, 0.25, 0.25], projection='polar')
    ax1.grid(False)
    ax1.axis('off')
    ax1.pcolormesh(azimuthsR*np.pi/180.0, radii, valuesR, cmap=mycmap)

    # Label AoLP angles
    for ii in np.arange(-90, 91, 30):
        iirad = ii*np.pi/180
        ax1.plot((iirad, iirad), (rmax-0.03, rmax+0.00), color='k', ls='-')
        ax1.plot((iirad, iirad), (rmin-0.00, rmin+0.03), color='k', ls='-')
        labl = str(ii) + "$^\circ$"
        if np.absolute(ii) == 90:
            labl = "$\pm 90^\circ$"
        ax1.text(iirad, 1.20, labl, style='italic', fontsize=12, rotation=0,
                 horizontalalignment='center', verticalalignment='center')

    pdf.savefig(fig)
    plt.close()

    d = pdf.infodict()
    d['Title'] = 'Polarization results'
    d['Author'] = 'Wheaton Physics and Astronomy'

print('All done.')
