import torch
import numpy as np
def himmelblau(x):
    return (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2

x=torch.tensor([0.,0.,],requires_grad=True)
optimizer = torch.optim.Adam([x],lr=1e-3)
##optimizer的含义： 每一个step x更新为 x = x - lr * ▲x
##  通过x requires_grad保留x的梯度 ，pred.backward()获取梯度
for step in range(200):
    pred = himmelblau(x)
    optimizer.zero_grad()
    pred.backward()
    optimizer.step()
    if step%2000 ==0:
        print('step{}:x={},f(x)={}'.format(step,x.tolist(),pred.item()))

##理解backward() requires_grad=True 保留梯度 backward 获取梯度值
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = x**3+y
z.backward()
print(z.item(), x.grad.item(), y.grad.item())
