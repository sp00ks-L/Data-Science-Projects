######################################################################################################################
#                                                                                                                    #
#             This is a collection of scripts that aim to investigate the movement data in the NFL dataset           #
#                                                                                                                    #
######################################################################################################################



import pandas as pd
import numpy as np
from scipy import maximum
import scipy.stats
import re

from math import sqrt, log10, log
from random import sample
from scipy.stats import boxcox, bartlett

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from matplotlib import patches, artist

sns.set('poster')
sns.set_style('darkgrid')
sns.set_palette('Set1')
plt.style.use('seaborn-poster')


injury_record = pd.read_csv("InjuryRecord.csv")
play_list = pd.read_csv("PlayList.csv")
# track_data = pd.read_csv("PlayerTrackData.csv")
# track_data_s = track_data.sample(frac=0.2)
p_key = [key for key in injury_record['PlayerKey'].unique()]
injured_p = play_list.loc[play_list['PlayerKey'].isin(p_key)]
# # game_1 = injured_p.loc[injured_p['GameID'] == '47813-11']
injury_record = injury_record.dropna().reset_index()
inj_record = [key for key in injury_record['PlayKey'].astype(str)]

injured_synthetic = injury_record['PlayKey'].loc[injury_record['Surface'] == 'Synthetic']
injured_natural = injury_record['PlayKey'].loc[injury_record['Surface'] == 'Natural']
inj_synth = [str(key) for key in injured_synthetic]
inj_nat = [str(key) for key in injured_natural]


######################################################
#             Creating animated line graphs          #
#             to understand player movement          #
######################################################

# for key in inj_synth:
#     plt.cla()
#     plt.clf()
#     plt.close()
# p1 = pd.read_excel(key + ".xlsx")
# x_coords = [x for x in p1['x']]
# y_coords = [y for y in p1['y']]
# # fig, ax = plt.subplots()
# # line, = ax.plot(x_coords, y_coords, color='k')
#
# speed = []
# # Distance formula / time
# """
# d = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
# I will do distance / second
# This will mean that 10 individual points have to be summed and divided by time (i.e 1)
# for now i will do every point / time which is 0.1 seconds
# """
# distance = []
# for i in range(len(x_coords) - 1):
#     sq_x = (x_coords[i + 1] - x_coords[i]) ** 2
#     sq_y = (y_coords[i + 1] - y_coords[i]) ** 2
#     distance.append(sqrt(sq_x + sq_y))
#
# speed = [round(dist / 0.1, 1) for dist in distance]
# speed.append(0)

# speed_template = 'Speed: %.1f yards/s'
# speed_avg_template = 'Max speed: %.1f yards/s'
# speed_text = ax.text(.07, .95, '', transform=ax.transAxes)
# speed_avg = ax.text(.07, .9, '', transform=ax.transAxes)

# def init():
#     line.set_data([], [])
#     speed_text.set_text('')
#     return line, speed_text
#
#
# def update(num, x, y, line):
#     line.set_data(x[:num], y[:num])
#     speed_text.set_text(speed_template % (speed[num]))
#     speed_avg.set_text(speed_avg_template % (np.max(speed)))
#     line.axes.axis([-10, 130, -10, 63])
#     end = patches.Rectangle((0, 0), 120, 53, linewidth=1, edgecolor='r', facecolor='none')
#     pitch = patches.Rectangle((10, 0), 100, 53, linewidth=1, edgecolor='g', facecolor='none')
#     # Add the patch to the Axes
#     ax.add_patch(end)
#     ax.add_patch(pitch)
#     print("{}%".format(round((num / len(x_coords)) * 100)))
#
#     return line, speed_text
#
#
# ani = animation.FuncAnimation(fig, update, len(x_coords), fargs=[x_coords, y_coords, line],
#                               interval=25, blit=True, init_func=init)
#
# writer = FFMpegWriter(fps=30, metadata=dict(artist='Synthetic'), bitrate=1800)
# ani.save(key + " (synthetic).mp4", writer=writer)


######################################################
#             Extracting speed data from             #
#                   player movement                  #
######################################################

"""
What info to extract
- injury type
- surface type
- Need to get top speed per surface type (maybe check how weather effects this)
"""
synth_speed = []
nat_speed = []

for key in inj_synth:
    p1 = pd.read_excel(key + ".xlsx")
    x_coords = [x for x in p1['x']]
    y_coords = [y for y in p1['y']]
    # fig, ax = plt.subplots()
    # line, = ax.plot(x_coords, y_coords, color='k')

    # Distance formula / time
    """
    d = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    I will do distance / second
    This will mean that 10 individual points have to be summed and divided by time (i.e 1)
    for now i will do every point / time which is 0.1 seconds
    """
    distance = []
    for i in range(len(x_coords) - 1):
        sq_x = (x_coords[i + 1] - x_coords[i]) ** 2
        sq_y = (y_coords[i + 1] - y_coords[i]) ** 2
        distance.append(sqrt(sq_x + sq_y))
        synth_speed.append(round((sqrt(sq_x + sq_y)) / 0.1, 1))

    # speed = [round(dist / 0.1, 1) for dist in distance]

for key in inj_nat:
    p1 = pd.read_excel(key + ".xlsx")
    x_coords = [x for x in p1['x']]
    y_coords = [y for y in p1['y']]
    # fig, ax = plt.subplots()
    # line, = ax.plot(x_coords, y_coords, color='k')

    # Distance formula / time
    distance = []
    for i in range(len(x_coords) - 1):
        sq_x = (x_coords[i + 1] - x_coords[i]) ** 2
        sq_y = (y_coords[i + 1] - y_coords[i]) ** 2
        distance.append(sqrt(sq_x + sq_y))
        nat_speed.append(round((sqrt(sq_x + sq_y)) / 0.1, 1))

"""
Average walking speed is 1.4 m/s. 
Filtering out when players are likely walking to clean up plots
"""
synthetic_spd = [spd for spd in synth_speed if spd > 1.5]
natural_spd = [spd for spd in nat_speed if spd > 1.5]


######################################################
#           Looking to see if any differences        #
#            where statistically significant         #
######################################################

# Samples pop = 4190, conf=99 err=3
# Sample size: 1283


synth_data = sample(synthetic_spd, 1283)
nat_data = sample(natural_spd, 1283)

df = pd.DataFrame({
    'synth_speed': synth_data,
    'natural_speed': nat_data
})

"""
Violin plot example to look at distribution
"""
# yrge = np.arange(3, 13, 1)
# xrge = np.arange(0, 2, 1)
# plt.figure(figsize=(10, 7))
# sns.violinplot(data=df, scale='area', saturation=0.7)
# plt.title("Speeds > 3.0 During the Injury Play \nSample Size: 1283")
# plt.xticks(xrge, labels=['Synthetic', 'Natural'])
# plt.axhline(np.mean(synth_data), 0.245, 0.255, color='white', aa=True)
# plt.axhline(np.mean(nat_data), 0.745, 0.755, color='white', aa=True)
# plt.ylabel("Speed (Yd/s)")
# plt.xlabel("Surface Type")
# plt.yticks(yrge)
# plt.tight_layout()
# plt.savefig("Violin - Over 3.png")
# plt.show()


synth_norm = [num / 1283 for num in synth_data]
nat_norm = [num / 1283 for num in nat_data]
result = scipy.stats.ttest_ind(synth_norm, nat_norm, equal_var=False)

sns.distplot(synth_norm)
sns.distplot(nat_norm)
plt.title("Box Cox")
plt.xlabel("Normalised Speed (YDs / s)")
# plt.annotate("TTest: {}".format(result[0]), (0.00175, 1380))
# plt.annotate("P-Value: {}".format(result[1]), (0.00175, 800))
plt.legend(['Synthetic', 'Natural'])
plt.savefig("Box Cox Over 1.5.png")
# plt.show()



