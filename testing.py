import numpy as np
from pytens import TensorNetwork
from scripts.svdinstn_decomposition import *
import matplotlib.pyplot as plt

# Initialize a numpy array
tens = np.random.randn(12, 15, 9)

# # Initialize a Tensor network
# tens = TensorNetwork()

# tens.add_node(0, Tensor(np.random.randn(3, 2),
#                          [Index(0, 3), Index(1, 2)]))
# tens.add_node(1, Tensor(np.random.randn(2, 8, 3),
#                          [Index(1, 2), Index(2, 8), Index(3, 3)]))
# tens.add_node(2, Tensor(np.random.randn(3, 6),
#                          [Index(3, 3), Index(4, 6)]))

# tens.add_edge(0, 1)
# tens.add_edge(1, 2)


tnet = FCTN(tens, 10, 1e-3)
tnet.initialize()
tnet.decompose()
# print([s.shape for s in tnet.G.values()])
# print([s.shape for s in tnet.S.values()])
tnet.to_tensor_network()
print([tnet.tn.value(key).shape for key in tnet.tn.network.nodes()])
print(tnet.stats)

tnet.tn.draw()
plt.show()
