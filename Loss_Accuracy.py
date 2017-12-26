import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = os.path.dirname(__file__)
modelname = 'Classifier'
var = 'var4'
file_path = os.path.join(file_path,modelname)
fig = plt.figure(figsize=(9,6))
plt.subplots_adjust(hspace = 0.5)

ax1 = fig.add_subplot(2,1,1)
ax1.set_title("Train")
train_pd = pd.read_csv(os.path.join(file_path,var,"data","train","train.csv"),
                        sep=',',
                        index_col='Iteration')

train_pd.plot(ax=ax1, grid=True)

ax2 = fig.add_subplot(2,1,2)
ax2.set_title("Validation")

val_pd = pd.read_csv(os.path.join(file_path,var,"data","val","val.csv"),
                       sep=',',
                       index_col='Iteration')

val_pd.plot(ax=ax2, grid=True)

namefig = "LossAccuracy_{0}.png".format(modelname)
plt.savefig(os.path.join(file_path,var,namefig))
