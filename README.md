# Polarizer Orientation Subimage Extraction, Interpolation, and Polarization Analysis

## Table of Contents

* [Overview](#overview)
* [Installation](#installation)
* [Purpose](#purpose)
* [Functionality](#functionality)
* [Usage](#usage)
* [Requirements](#requirements)
* [Output](#output)
* [Notes](#notes)
* [Acknowledgments](#acknowledgments)
* [License](#license)

## Overview

This Python script extracts subimages for different polarizer orientations from a raw image file captured by a polarization-sensitive sensor. The code Interpolates the subimages, calculates the Degree of Linear Polarization (DoLP), Angle of Linear Polarization (AoLP), and visualizes the results in a PDF report.

## Installation

1. Ensure you have Python 3.x installed on your system.
2. Install required packages using pip:

```bash
pip install numpy matplotlib astropy
```

## Purpose

This Python script processes raw image data captured by a polarization-sensitive sensor, extracting subimages for different polarizer orientations. It calculates the Degree of Linear Polarization (DoLP) and Angle of Linear Polarization (AoLP) for the images and visualizes the results in a PDF report. Researchers and practitioners can use this script to analyze and interpret polarization properties in various fields, such as remote sensing, biomedical imaging, and materials characterization. The script provides a convenient and informative solution for studying polarized light phenomena in raw image data.

## Functionality

1. Reads the raw image data from a specified file and decodes it based on its format (Mono8 or Mono12).
2. Divides the image into four subimages, corresponding to pixel orientations of 0, 45, 90, and 135 degrees.
3. Interpolates any missing (NaN) values in the subimages to obtain smoother results.
4. Calculates the DoLP and AoLP from the interpolated subimages.
5. Visualizes the original image, interpolated subimages, DoLP, and AoLP in separate plots.
6. Saves the visualizations in a PDF report.

## Usage

1. Before running the code, modify the `datadir` variable in the script to point to the folder containing the raw image file.
2. Also, update the `ipfile` variable to specify the name of the raw image file (e.g., `asun20ms.raw`).
3. Run the script to extract subimages, perform polarization analysis, and generate the PDF report.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Astropy


## Output

If the script runs without any errors, it will produce a PDF report (<ipfile>.pdf) containing the following visualizations:

* Raw image from the sensor.
* Interpolated subimages for polarizer orientations of 0, 45, 90, and 135 degrees.
* Degree of Linear Polarization (DoLP) map.
* Angle of Linear Polarization (AoLP) map along with a half-circular color bar.

## Notes

* The script currently supports raw images in either Mono8 or Mono12 format.
* Ensure the raw image dimensions match the specified img_height and img_width in the code.

## Acknowledgments

This code was developed by Sayed Ibrahimi as part of MEASURING THE POLARIZATION OF THE SOLAR CORONA
DURING THE TOTAL SOLAR ECLIPSE OF APRIL 8, 2024 led by Professor Dipankar Maitra at Wheaton College MA.

## License

This project is licensed under the MIT License.
