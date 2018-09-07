import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('KNN','SVM','Random Forest','Bi-LSTM')
y_pos = np.arange(len(objects))
performance = [48.38,55.45,52.99,64.66]

plt.bar(y_pos, performance, align='center', alpha=0.5, color= 'Blue')
plt.xticks(y_pos, objects,rotation = 'vertical')
plt.ylabel('Accuracy')
plt.title('Performance of Classifiers')
plt.ylim(0,100)
plt.show()
