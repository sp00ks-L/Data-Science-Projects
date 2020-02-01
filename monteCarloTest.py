import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('bmh')

"""
ggplot
dark_background
bmh
classic
tableau-colorblind10

"""

funds_list = []
fund_totals = []


def dice_roll():
    roll = random.randint(1, 100)
    if roll <= 51:
        return False
    elif 51 < roll < 100:
        return True
    elif roll == 100:
        return False


def simple_bettor(funds, initial_wager, wager_count):
    value = funds
    wager = initial_wager
    bettor_funds = []

    no_wagers = 0

    while no_wagers < wager_count:
        if dice_roll():
            value += wager
        else:
            value -= wager
        bettor_funds.append(value)
        no_wagers += 1
    funds_list.append(bettor_funds)
    fund_totals.append(value)


x = 0
iterations = 1000
while x < iterations:
    simple_bettor(10000, 100, 10000)
    x += 1

for l in funds_list:
    plt.plot(l)

plt.xlabel("Bet Number")
plt.ylabel("Funds")
plt.minorticks_on()
plt.title("Simple Betting Monte Carlo")
plt.show()

prof_count = 0
loss_count = 0
even_count = 0
for val in fund_totals:
    if val < 10000:
        loss_count += 1
    elif val > 10000:
        prof_count += 1
    else:
        even_count += 1


def find_percent(bettors):
    prof = prof_count
    loss = loss_count
    even = even_count

    prof_perc = (prof / bettors) * 100
    loss_perc = (loss / bettors) * 100
    even_perc = (even / bettors) * 100

    print("Profits: {0}% \nLosses: {1}% \nBreak Even: {2:.2f}%".format(prof_perc, loss_perc, even_perc))


find_percent(iterations)
