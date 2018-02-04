#Thompson sampling py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as random

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
N = 10000
d = 10

ads_selected = []
number_of_rewards_1 = [0]*d
number_of_rewards_0 = [0]*d
total_reward = 0
for n in range(0, N):
    max_random = 0
    ad_choice = 0
    for i in range(0,d):
        random_beta = random.betavariate(number_of_rewards_1[i]+1, number_of_rewards_0[i]+1)
        if random_beta > max_random:
            max_random = random_beta
            ad_choice = i
    ads_selected.append(ad_choice)
    reward = dataset.values[n, ad_choice]
    
    if(reward==1):
        number_of_rewards_1[ad_choice] =  number_of_rewards_1[ad_choice] + 1
    else:
        number_of_rewards_0[ad_choice] =  number_of_rewards_0[ad_choice] + 1

    total_reward = total_reward + reward
    
    
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()