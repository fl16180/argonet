#!/usr/bin/env python

''' quickplot.py
    Plotting utility taking an output csv file with predictions from argo-system
    and producing a simple time series plot, with the option of saving the plot
    as a png.

    Usage on Unix:
        ./quickplot.py inputfilename.csv (-s outputfilename.png)
    Usage on Windows (works on all systems):
        python quickplot.py inputfilename.csv (-s outputfilename.png)

Fred Lu
'''

import matplotlib.pyplot as plt
import pandas as pd
import sys


# check command line arguments
if len(sys.argv) == 2:
    print 'Loading', sys.argv[1]
    _save = False
elif (len(sys.argv) == 4) and (sys.argv[2] == '-s'):
    print 'Loading', sys.argv[1]
    _save = True
else:
    print 'Usage: python quickplot.py inputfilename.csv (-s outputfilename.png)'
    sys.exit()

# load csv file
infile = pd.read_csv(sys.argv[1], parse_dates=[0])

# initialize figure space
fig = plt.figure()
ax = plt.subplot(3, 1, (1, 2))

# extract date column
time = infile.iloc[:, 0].values

# iterate over other columns to plot
lines = []
for i in range(1, infile.shape[1]):
    _name = list(infile)[i]
    _lw = 2.5 if _name == 'target' else 1.5
    lines.append(ax.plot_date(x=time, y=infile.iloc[:, i].values, fmt='-', label=_name, linewidth=_lw))

# plot settings
ax.legend()
plt.grid()
plt.title('Quickplot for {0}'.format(sys.argv[1]))
plt.ylabel('% ILI')
fig = plt.gcf()
fig.set_size_inches(12, 10)
plt.show()

# save file if specified
if _save is True:
    fig.savefig(sys.argv[3], format='png')
