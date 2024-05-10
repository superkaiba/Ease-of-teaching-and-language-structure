import torch
import numpy as np
import pdb
fpath = "/home/mila/t/thomas.jiralerspong/kolmogorov/scratch/kolmogorov/ease_of_teaching/results/new/recreate_original_no_reset/May_06_2024_15:45:52/i=250000_topo_measure=0.295037968700386_accuracy=0.89_z.pt"

z = torch.load(fpath)
batch_size, length = z.shape
z_new = np.zeros((batch_size, 2))

z_new[:, 0] = np.argmax(z[:, 0:8], axis=1)
z_new[:, 1] = np.argmax(z[:, 8:16], axis=1)
# idxes = []
# for end in range(10, 41, 10):
#     start = end - 10
#     idx = np.argmax(z[:, start:end], axis=1)
#     z_new[:, start//10] = idx

z_new = torch.tensor(z_new, dtype=torch.int)
torch.save(z_new, fpath.replace("z", "z_new"))


        



