import numpy as np
keep_prob=0.7
a3=np.random.randint(1,size=(3,1),high=10)
print(a3)
print(a3.shape[0])
print(a3.shape[1])
d3=np.random.rand(a3.shape[0],a3.shape[1])<keep_prob
a3*=d3
print(d3)
print(a3)
a3=a3/0.7
print(a3)