import keepsake
import os
import matplotlib.pyplot as plt


nn_exp = keepsake.Project(repository="file://keepsake-repository-reprodl")

nn_exp.experiments.list().scatter(param="learning_rate", metric="train_loss")
plt.savefig("train_loss_vs_lr.pdf", bbox_inches = "tight")

plt.close()

nn_exp.experiments.list().plot()
plt.savefig("train_loss.pdf", bbox_inches = "tight")