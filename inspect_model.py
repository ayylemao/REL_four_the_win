'''
Inspect learnt convolution kernels.
'''
import torch as th
import matplotlib.pyplot as plt

model = th.load("model.torch")

# kernels of first conv layer
nrows = 10
ncols = 10
fig, axs = plt.subplots(nrows, ncols)
kernels = []
counter = 0
for i in range(nrows):
    for j in range(ncols):
        kernel = model.get("model_state_dict").get("conv1.weight")[counter].squeeze().squeeze()
        axs[i,j].imshow(kernel)
        kernels.append(kernel)
        counter += 1
plt.show()

