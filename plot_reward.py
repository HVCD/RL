import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_name = "mean.csv"
data = pd.read_csv(csv_name)
data = np.array(data)
data1 = pd.read_csv('sigma.csv')
data1 = np.array(data1)


plt.plot(data[:, 1], c='b', label='mean')
plt.figure(2)
plt.plot(data1[:, 1], c='g', label='sigma')
#plt.xlabel('Train Step')

#plt.xticks(data[:, 0], ('$3M$', '$3.5M$', '$4.0M$', '$4.5M$'), fontsize=5)
#plt.title('Compare Loss of LSTM-RNN and Basic-RNN')
#plt.legend(loc='best', prop={'size': 22})
#plt.ylim([300, 450])
#plt.ylabel('Loss')
plt.legend()
plt.show()
