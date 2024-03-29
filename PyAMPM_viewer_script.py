#############################################################################
#############################################################################
# Code description
"""PyAMPM viewer is an AMPM file reader script."""

# Author
__author__ = 'Samy Hocine'

#############################################################################
#############################################################################
# Modules & libraries
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

#############################################################################
#############################################################################
# Variables

# Define the default location as where the code is executed
__location__ = os.path.dirname(os.path.realpath(__file__))

u8  = np.uint8
u16 = np.uint16
u32 = np.uint32
u64 = np.uint64
                                    # Python index
ChannelNames = ['GalvoXDemandBits', # 0
    'GalvoXDemandCartesian',        # 1
    'GalvoYDemandBits',             # 2
    'GalvoYDemandCartesian',        # 3
    'FocusDemandBits',              # 4
    'FocusDemandCartesian',         # 5
    'GalvoXActualBits',             # 6
    'GalvoXActualCartesian',        # 7
    'GalvoYActualBits',             # 8
    'GalvoYActualCartesian',        # 9
    'FocusActualBits',              # 10
    'FocusActualCartesian',         # 11
    'Modulate',                     # 12
    'BeamDumpDiodeBits',            # 13
    'BeamDumpDiodeWatts',           # 14
    'Photodiode1Bits',              # 15
    'Photodiode1Watts',             # 16
    'Photodiode2Bits',              # 17
    'Photodiode2Watts',             # 18
    'PSDPositionXBits',             # 19
    'PSDPositionYBits',             # 20
    'PSDIntensity',                 # 21
    'PowerValue1',                  # 22
    'PowerValue2',                  # 23
    'Photodiode1Normalised',        # 24
    'Photodiode2Normalised',        # 25
    'BeamDumpDiodeNormalised',      # 26
    'LaserBackReflection',          # 27
    'OutputPowerMonitor']           # 28

#############################################################################
#############################################################################
# Functions

def read(datHandle, dtype, count=-1):
    """Read data from binary file."""
    return 1.0*np.fromfile(datHandle, dtype=dtype, count=count)

def readInt(datHandle, dtype):
    """Return data as integer from binary file."""
    return read(datHandle, dtype, 1)[0].astype('int')

def read100Kdata(datHandle, dtype, numChannels, numSamples, channelIDs):
    """Read data for 100K AMPM files."""
    # Read the raw data from binary file and reshape into array for each channel
    rawData_array = np.fromfile(datHandle, dtype=dtype, count=numChannels*numSamples)
    rawData = np.reshape(rawData_array,[-1,numChannels]).transpose()

    # Build the time series plots
    # numChannels and numSamples are inverted because of the transpose
    maxChannels = channelIDs[numChannels-1].astype('int') + 1
    tSeries = np.zeros([maxChannels,numSamples], dtype=float)

    for ind in range(numChannels):
        channel = channelIDs[ind].astype('int')
        tSeries[channel] = rawData[ind].astype('float')

    # turn cartesian data from unsigned to signed
    for i in range(1,12,2):
        tSeries[i] = np.where(tSeries[i] > 32768, tSeries[i] - 65535, tSeries[i])

    # adjust for pipeline delay
    CircShift = [-44, -42, -45]

    # shift the modulate by 20usec to accound for pipeline delay.
    data_roll = [range(6,12),range(12,13),range(13,29)]
    for n in range(3):
        for i in data_roll[n]:
            tSeries[i] = np.roll(tSeries[i],CircShift[n])

    # Convert Galvo Demand and Actual X Y Z to millimeters
    for i in range(1,12,2):
        tSeries[i] = np.multiply(tSeries[i],0.005)

    # Smooth the data
    # Filter the Galvo feedback data
    filterwidth = 21
    filter = np.ones(filterwidth)
    filter = np.divide(filter,filterwidth)

    for i in range(7,10,2):
        tSeries[i] = np.convolve(tSeries[i], filter, 'same')

    tSeries = np.delete(tSeries,range(numSamples-(filterwidth+1),numSamples),1)
    tSeries = np.delete(tSeries,range(filterwidth),1)

    return tSeries, filterwidth

# Code from Renishaw
def read2Mdata(datHandle, dtype, count=-1):
    """Read data for 2M AMPM files."""
    a = np.fromfile(datHandle, dtype=dtype, count=count)
    a.byteswap(True)
    z = np.zeros(len(a)+1, dtype=dtype)

    lenOdd = len(a[1::2])
    lenEven = len(a[::2])

    z[::2][:lenOdd] = a[1::2]
    z[1::2][:lenEven] = a[::2]

    z = z[:-1]

    if len(a)%2!=0:
        z[-1] = a[-1]
    return 1.0*z

def readAMPMdat(datFolder, datFile): # Equivalent of ReadAMPMDataFast on Matlab
    """Read the AMPM dat file and build the time series array."""
    # build the complete path to the datfile
    datPath = datFolder / datFile

    # open with rb statement to read binary file format
    with open(datPath, 'rb') as datHandle:
        # version       = readInt(datHandle, u32) # written from Renishaw python code
        version_major   = readInt(datHandle, u16) # Renishaw Matlab code
        version_minor   = readInt(datHandle, u16) # Renishaw Matlab code
        laser           = readInt(datHandle, u8)
        installedLasers = readInt(datHandle, u8)
        initialTime     = readInt(datHandle, u64)
        layerID         = readInt(datHandle, u32)
        layerHeight     = readInt(datHandle, u32) # Should be height in microns
        pipelineDelay   = readInt(datHandle, u16)
        sampleFreq      = readInt(datHandle, u32)
        numSamples      = readInt(datHandle, u32)
        numChannels     = readInt(datHandle, u16)
        channelIDs      = read(datHandle, u8, 50)

                                    # Python index
        hdrs = [version_major,      # 0
                version_minor,      # 1
                laser,              # 2
                installedLasers,    # 3
                initialTime,        # 4
                layerID,            # 5
                layerHeight,        # 6
                pipelineDelay,      # 7
                sampleFreq,         # 8
                numSamples,         # 9
                numChannels,        # 10
                channelIDs]         # 11

        tEnd = numSamples/sampleFreq
        t = np.linspace(0, tEnd, numSamples, endpoint=False)

        # for 100K samples
        if sampleFreq == 100000:
            data, filterwidth = read100Kdata(datHandle, u16, numChannels, numSamples, channelIDs)
            t = np.delete(t,range(numSamples-(2*filterwidth+1),numSamples),0)

        # for 2M samples
        if sampleFreq == 2000000:
            numChannels = 1
            # Regardless of what the header might say, there is only one
            # channel in a 2 MHz file
            data = read2Mdata(datHandle, u16, numChannels*numSamples)

    print(f'Data read for file {datFile}')
    return t, data, sampleFreq


#############################################################################
#############################################################################
# Main body

# Default path
# datFolder = Path(__location__)

# Custom path (uncomment)
custom_path = 'C:/Users/rlamb/Dropbox (UCL)/BeamtimeData/ME-1573 - ESRF ID19/ME1573_AMPM_data'
datFolder = Path(custom_path)

# File name to load
datFile = '0103_AMPM_1_L4_100K_2022-01-26_06-22-31.dat'

ft, fData, fsampleFreq = readAMPMdat(datFolder, datFile)

# Trim to region of interest
# for i, val in enumerate(fData[15]):
    # start = 0
    # if val > 100:
        # start = i - 20
        # break

fig, ax = plt.subplots(figsize=(16,9))

if fsampleFreq == 100000: # For 100K files
    ind = 15 #Channel index (see above)
    ax.plot(ft, fData[ind])
    # ax.plot(np.subtract(ft[start:], ft[start]), fData[ind][start:])
    ax.set_ylabel(ChannelNames[ind])

if fsampleFreq == 2000000: # For 2M files
    ax.plot(ft, fData)
    ax.set_ylabel('BDPD [arbitrary units]')

ax.grid(True)
ax.set_title(datFile)
ax.set_xlabel('Time [s]')

plt.show()

