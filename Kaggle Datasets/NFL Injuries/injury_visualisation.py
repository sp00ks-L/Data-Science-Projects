import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set('poster')
sns.set_style('darkgrid')
plt.style.use('seaborn-colorblind')
# plt.style.use('ggplot')

injury_record = pd.read_csv("InjuryRecord.csv")
play_list = pd.read_csv("PlayList.csv")
# player_track = pd.read_csv("PlayerTrackData.csv")

######################################################
#               Types of injuries graph              #
######################################################
# ax = injury_record['BodyPart'].value_counts().plot(kind='barh', figsize=(10, 7),
#                                                    color="dodgerblue", fontsize=15)
#
# ax.set_alpha(0.8)
# ax.set_title("Types of Injuries Suffered\nTotal Injuries: 105", fontsize=18)
# ax.set_xlabel("Frequency of Injury", fontsize=18)
# ax.set_ylabel("Body Part Injured", fontsize=18)
# ax.set_xticks(np.arange(0, 56, 5))
#
# # create a list to collect the plt.patches data
# totals = []
#
# # find the values and append to list
# for i in ax.patches:
#     totals.append(i.get_width())
#
# # set individual bar labels using above list
# total = sum(totals)
#
# # set individual bar labels using above list
# for i in ax.patches:
#     # get_width pulls left or right; get_y pushes up or down
#     ax.text(i.get_width() + .2, i.get_y() + .23,
#             str(round((i.get_width() / total) * 100, 2)) + '%', fontsize=15,
#             color='dimgrey')
#
#
# # invert for largest on top
# ax.invert_yaxis()
# plt.tight_layout()
# plt.savefig("Injuries From Pandas (H).png")
# plt.show()

######################################################
#               Length of injuries graph             #
######################################################
# print(injury_record.iloc[0])
# 'DM_M42', 'DM_M28', 'DM_M7', 'DM_M1'
# dm42 = injury_record.loc[injury_record['DM_M42'] == 1]
# dm28 = injury_record.loc[injury_record['DM_M28'] == 1]
# dm7 = injury_record.loc[injury_record['DM_M7'] == 1]
# dm1 = injury_record.loc[injury_record['DM_M1'] == 1]
#
# d42, d28, d7, d1 = len(dm42), len(dm28), len(dm7), len(dm1)
#
# fig = plt.figure(figsize=(10, 7))
#
# ax = fig.add_subplot(111)
# fig.subplots_adjust(top=0.85)
# ax.set_title('Distribution of Injury Length')
# ax.set_xlabel('Length of Injury in Days', fontdict={'size': 14, 'weight': 'bold'})
# ax.set_ylabel('Number of Injuries', fontdict={'size': 14, 'weight': 'bold'})
#
# plt.bar(x=['>7', '>28', '>42'], height=[d7, d28, d42])
#
# # get_x pulls left or right; get_height pushes up or down
# ax.text(-0.08, 77, "{}%".format(round((76 / 105) * 100, 2)), color='dimgrey', fontsize=15)
# ax.text(0.92, 38, "{}%".format(round((37 / 105) * 100, 2)), color='dimgrey', fontsize=15)
# ax.text(1.92, 30, "{}%".format(round((29 / 105) * 100, 2)), color='dimgrey', fontsize=15)
# ax.text(1.365, 52.5, 'All injuries lasted for at least 1 day\nTotal injuries: 105', fontdict={'size': 15},
#         bbox={'facecolor': 'blue', 'ec': 'black', 'alpha': 0.3, 'pad': 10, 'lw': 2, 'capstyle': 'projecting'})
#
# plt.tight_layout()
# plt.savefig("Length of Injury.png")
# plt.show()


######################################################
#             Link between injury length +           #
#                 body part injured                  #
######################################################
# # Data
# fig = plt.figure(figsize=(20, 7))
# ax = fig.add_subplot(111)
# ax.set_title('Relative Recovery Time to Injury', fontdict={'size': 30, 'weight': 'bold'}, pad=20)
# ax.set_xlabel('Injured Area', fontdict={'size': 25, 'weight': 'bold'})
# ax.set_ylabel('Total Percentage of Injuries', fontdict={'size': 25, 'weight': 'bold'})
#
# raw_data = {'less_than_7': [11, 16, 0, 2, 0], 'greater_than_7': [21, 10, 0, 4, 1], 'greater_than_28': [3, 5, 2, 1, 0],
#             'greater_than_42': [13, 11, 5, 0, 0]}
# df = pd.DataFrame(raw_data)
#
# # From raw value to percentage
# totals = [i + j + k + l for i, j, k, l in
#           zip(df['less_than_7'], df['greater_than_7'], df['greater_than_28'], df['greater_than_42'])]
# less_than_7 = [i / j * 100 for i, j in zip(df['less_than_7'], totals)]
# greater_than_7 = [i / j * 100 for i, j in zip(df['greater_than_7'], totals)]
# greater_than_28 = [i / j * 100 for i, j in zip(df['greater_than_28'], totals)]
# greater_than_42 = [i / j * 100 for i, j in zip(df['greater_than_42'], totals)]
#
# barWidth = 0.5
# r = [0, .9, 1.8, 2.7, 3.6]
# p1 = plt.bar(r, less_than_7,
#              edgecolor='black',
#              width=barWidth,
#              linewidth=2)
# p2 = plt.bar(r, greater_than_7, bottom=less_than_7,
#              edgecolor='black',
#              width=barWidth,
#              linewidth=2)
# p3 = plt.bar(r, greater_than_28, bottom=[i + j for i, j in zip(less_than_7, greater_than_7)],
#              edgecolor='black',
#              width=barWidth,
#              linewidth=2)
# p4 = plt.bar(r, greater_than_42, bottom=[i + j + k for i, j, k in zip(less_than_7, greater_than_7, greater_than_28)],
#              edgecolor='black',
#              width=barWidth,
#              linewidth=2)
#
# # Percentage Annotations
# # Knee
# ax.text(-0.09, 85, "{}%".format(round((13 / 48) * 100, 2)), color='white', fontsize=15)
# ax.text(-0.09, 68.5, "{}%".format(round((3 / 48) * 100, 2)), color='white', fontsize=15)
# ax.text(-0.09, 43, "{}%".format(round((21 / 48) * 100, 2)), color='white', fontsize=15)
# ax.text(-0.09, 10, "{}%".format(round((11 / 48) * 100, 2)), color='white', fontsize=15)
#
# # Ankle
# ax.text(.81, 85, "{}%".format(round((11 / 42) * 100, 2)), color='white', fontsize=15)
# ax.text(.81, 67, "{}%".format(round((5 / 42) * 100, 2)), color='white', fontsize=15)
# ax.text(.81, 48, "{}%".format(round((10 / 42) * 100, 2)), color='white', fontsize=15)
# ax.text(.81, 19, "{}%".format(round((16 / 42) * 100, 2)), color='white', fontsize=15)
#
# # Foot
# ax.text(1.70, 63, "{}%".format(round((5 / 7) * 100, 2)), color='white', fontsize=15)
# ax.text(1.70, 16, "{}%".format(round((2 / 7) * 100, 2)), color='white', fontsize=15)
#
# # Toes
# ax.text(2.6, 90, "{}%".format(round((1 / 7) * 100, 2)), color='white', fontsize=15)
# ax.text(2.6, 60, "{}%".format(round((4 / 7) * 100, 2)), color='white', fontsize=15)
# ax.text(2.6, 17, "{}%".format(round((2 / 7) * 100, 2)), color='white', fontsize=15)
#
# # Heel
# ax.text(3.5, 50, "{}%".format(round((1 / 1) * 100, 2)), color='white', fontsize=15)
#
# yrge = np.arange(0, 101, 20)
# plt.xticks(r, labels=['Knee', 'Ankle', 'Foot', 'Toes', 'Heel'])
# plt.yticks(yrge, labels=['0%', '20%', '40%', '60%', '80%', '100%'])
# plt.legend((p4[0], p3[0], p2[0], p1[0]),
#            ('43+', '29 - 42', '7 - 28', '1- 6'), title='Number of Days', loc=7)
# plt.xlim(-.5, 5)
# fig.tight_layout()
# sns.despine(bottom=True, right=True, top=True, left=True)
# plt.savefig("Relative Recovery Time.png")
# plt.show()


######################################################
#             Link between injury length +           #
#           body part injured + surface type         #
######################################################
# Data
# print(injury_record.iloc[1])
# We want to filter: Body part and surface then check recovery time distributions
# Data
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111)

raw_data = {'lt7_synth': [6, 8, 0, 2, 0], 'lt7_nat': [5, 8, 0, 0, 0], 'gt7_synth': [9, 7, 0, 3, 0],
            'gt7_nat': [12, 6, 0, 1, 1], 'gt28_synth': [2, 2, 1, 1, 0], 'gt28_nat': [1, 0, 1, 0, 0],
            'gt42_synth': [7, 8, 1, 0, 0], 'gt42_nat': [6, 3, 4, 0, 0]}
df = pd.DataFrame(raw_data)

# From raw value to percentage
synth_totals = [a + b + c + d for a, b, c, d in
                zip(df['lt7_synth'], df['gt7_synth'], df['gt28_synth'], df['gt42_synth'])]
nat_totals = [e + f + g + h for e, f, g, h in
              zip(df['lt7_nat'], df['gt7_nat'], df['gt28_nat'], df['gt42_nat'])]

lt7_synth = [i / j * 100 for i, j in zip(df['lt7_synth'], synth_totals) if j != 0]
gt7_synth = [i / j * 100 for i, j in zip(df['gt7_synth'], synth_totals) if j != 0]
gt28_synth = [i / j * 100 for i, j in zip(df['gt28_synth'], synth_totals) if j != 0]
gt42_synth = [i / j * 100 for i, j in zip(df['gt42_synth'], synth_totals) if j != 0]

lt7_nat = [i / j * 100 for i, j in zip(df['lt7_nat'], nat_totals)]
gt7_nat = [i / j * 100 for i, j in zip(df['gt7_nat'], nat_totals)]
gt28_nat = [i / j * 100 for i, j in zip(df['gt28_nat'], nat_totals)]
gt42_nat = [i / j * 100 for i, j in zip(df['gt42_nat'], nat_totals)]


barWidth = 1.5
r = [0, 2, 4, 6]
p1 = plt.bar(r, lt7_synth,
             edgecolor='black',
             width=barWidth,
             linewidth=2)
p2 = plt.bar(r, gt7_synth, bottom=lt7_synth,
             edgecolor='black',
             width=barWidth,
             linewidth=2)
p3 = plt.bar(r, gt28_synth, bottom=[i + j for i, j in zip(lt7_synth, gt7_synth)],
             edgecolor='black',
             width=barWidth,
             linewidth=2)
p4 = plt.bar(r, gt42_synth, bottom=[i + j + k for i, j, k in zip(lt7_synth, gt7_synth, gt28_synth)],
             edgecolor='black',
             width=barWidth,
             linewidth=2)
"""
#0072b2 - blue
#009e73 - green
#d55e00 - orange
#cc79a7 - pink
"""

r2 = [10, 12, 14, 16, 18]
p5 = plt.bar(r2, lt7_nat,
             edgecolor='black',
             facecolor='#0072b2',
             width=barWidth,
             linewidth=2)
p6 = plt.bar(r2, gt7_nat, bottom=lt7_nat,
             edgecolor='black',
             facecolor='#009e73',
             width=barWidth,
             linewidth=2)
p7 = plt.bar(r2, gt28_nat, bottom=[i + j for i, j in zip(lt7_nat, gt7_nat)],
             edgecolor='black',
             facecolor='#d55e00',
             width=barWidth,
             linewidth=2)
p8 = plt.bar(r2, gt42_nat, bottom=[i + j + k for i, j, k in zip(lt7_nat, gt7_nat, gt28_nat)],
             edgecolor='black',
             facecolor='#cc79a7',
             width=barWidth,
             linewidth=2)

yrge = np.arange(0, 101, 20)
xrge = np.arange(0, 20, 1)
ax.text(-.5, 113, "Relative Recovery Time Grouped by Field Type", fontdict={'size': 25, 'weight': 'bold'})
ax.text(1.7, 103, "Synthetic\n57 Injuries", fontdict={'size': 18})
ax.text(13.1, 103, "  Natural\n48 Injuries", fontdict={'size': 18})
ax.text(5.5, -29, 'Injured Area', fontdict={'size': 25, 'weight': 'bold'})
ax.set_ylabel('Total Percentage of Injuries', fontdict={'size': 20, 'weight': 'bold'})
plt.yticks(yrge, labels=['0%', '20%', '40%', '60%', '80%', '100%'])
plt.xticks(xrge,
           labels=['Knee', "", 'Ankle', "", 'Foot', "", 'Toes', "", "", "", 'Knee', "", 'Ankle', "", 'Foot', "", 'Toes',
                   "", 'Heel'], rotation=30)

# Percentage Annotations
# Knee Synth
ax.text(-.4, 85, "{}%".format(round((7 / 24) * 100)), color='white', fontsize=15)
ax.text(-.28, 65.3, "{}%".format(round((2 / 24) * 100)), color='white', fontsize=15)
ax.text(-.4, 43, "{}%".format(round((9 / 24) * 100)), color='white', fontsize=15)
ax.text(-.4, 10, "{}%".format(round((6 / 24) * 100)), color='white', fontsize=15)

# Ankle Synth
ax.text(1.6, 85, "{}%".format(round((8 / 25) * 100)), color='white', fontsize=15)
ax.text(1.72, 62.3, "{}%".format(round((2 / 25) * 100)), color='white', fontsize=15)
ax.text(1.6, 43, "{}%".format(round((7 / 25) * 100)), color='white', fontsize=15)
ax.text(1.6, 15, "{}%".format(round((8 / 25) * 100)), color='white', fontsize=15)

# Foot Synth
ax.text(3.6, 73, "{}%".format(round((1 / 2) * 100)), color='white', fontsize=15)
ax.text(3.6, 27, "{}%".format(round((1 / 2) * 100)), color='white', fontsize=15)

# Toes Synth
ax.text(5.6, 90, "{}%".format(round((1 / 6) * 100)), color='white', fontsize=15)
ax.text(5.6, 59, "{}%".format(round((3 / 6) * 100)), color='white', fontsize=15)
ax.text(5.6, 17, "{}%".format(round((2 / 6) * 100)), color='white', fontsize=15)

# Natural ----------------

# Knee Nat
ax.text(9.6, 85, "{}%".format(round((6 / 24) * 100)), color='white', fontsize=15)
ax.text(9.72, 71.5, "{}%".format(round((1 / 24) * 100)), color='white', fontsize=14)
ax.text(9.6, 45, "{}%".format(round((12 / 24) * 100)), color='white', fontsize=15)
ax.text(9.6, 10, "{}%".format(round((5 / 24) * 100)), color='white', fontsize=15)

# Ankle Nat
ax.text(11.6, 89, "{}%".format(round((3 / 17) * 100)), color='white', fontsize=15)
ax.text(11.6, 63, "{}%".format(round((6 / 17) * 100)), color='white', fontsize=15)
ax.text(11.6, 25, "{}%".format(round((8 / 17) * 100)), color='white', fontsize=15)

# Foot Bat
ax.text(13.6, 60, "{}%".format(round((4 / 5) * 100)), color='white', fontsize=15)
ax.text(13.6, 10, "{}%".format(round((1 / 5) * 100)), color='white', fontsize=15)

# Toes Bat
ax.text(15.5, 50, "{}%".format(round((1 / 1) * 100)), color='white', fontsize=14)

# Heel Nat
ax.text(17.5, 50, "{}%".format(round((1 / 1) * 100)), color='white', fontsize=14)


# plt.suptitle("Relative Recovery Time")
plt.legend((p4[0], p3[0], p2[0], p1[0]),
           ('43+', '29 - 42', '7 - 28', '1- 6'), title='Number of Days', fontsize=11, title_fontsize=14)
plt.xlim(-1, 23)
plt.ylim(-1, 101)
fig.tight_layout()
sns.despine(bottom=True, right=True, top=True, left=True)
plt.savefig("Surface Recovery.png")
plt.show()
