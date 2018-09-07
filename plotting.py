import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = {'Features':['Attention','Meditation','Raw','Delta','Theta','Aplha1','Aplha2','Beta1','Beta2','Gamma1','Gamma2'],
        #'KNN':[47.65,48.19,47.67,48.01,49.12,48.17,48.14,48.73,48.24,48.26,48.09],
         #'Random Forest':[51.47,53.03,53.12,52.39,52.55,53.15,53.00,53.95,52.89,53.13,53.12],
        'Bidirectional LSTM':[63.01,64.37,63.43,64.02,64.09,64.07,63.85,63.68,63.65,63.67,63.55]}
#df = pd.DataFrame(data, columns = ['Features', 'KNN', 'Random Forest', 'Bidirectional LSTM'])
df = pd.DataFrame(data, columns = ['Features', 'Bidirectional LSTM'])

print(df)

pos = list(range(len(df['Features'])))
width = 0.50
fig, ax = plt.subplots(figsize=(10,5))

plt.bar(pos,
        df['Bidirectional LSTM'],
        width,
        alpha=0.5,
        color='#EE3224',
        label='KNN')
'''
plt.bar([p + width for p in pos],
        df['Random Forest'],
        width,
        alpha=0.5,
        color='#2C3221',
        label='Random Forest')
plt.bar([p + width*2  for p in pos],
        df['Bidirectional LSTM'],
        width,
        alpha=0.5,
        color='#F78F1E',
        label='Bidirectional LSTM')
'''
ax.set_ylabel('Accuracy')

# Set the chart's title
ax.set_title('Leave one Feature out')

# Set the position of the x ticks
ax.set_xticks([p + 1.5 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(df['Features'])

# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*4)
plt.ylim([0, 100])

# Adding the legend and showing the plot
plt.legend(['Bidirectional LSTM'], loc='upper left')
plt.grid()
plt.show()
