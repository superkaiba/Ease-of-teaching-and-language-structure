import torch
import numpy as np
fpath = "/home/mila/t/thomas.jiralerspong/kolmogorov/scratch/kolmogorov/results/to_prequential_code/with_reset_topo=0.234_acc=84/z_old.pt"

z = torch.load(fpath)
batch_size, length = z.shape
z_new = np.zeros((batch_size, 4))

idxes = []
for end in range(10, 41, 10):
    start = end - 10
    idx = np.argmax(z[:, start:end], axis=1)
    z_new[:, start//10] = idx

z_new = torch.tensor(z_new, dtype=int)
torch.save(z_new, fpath.replace("z_old", "z"))


        



