import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from scheduler_warmup import CosineAnnealingLR_Warmup

net = nn.Sequential(nn.Linear(1,1,bias=False))
optimizer = torch.optim.SGD(net.parameters(),lr=1e-3)
scheduler = CosineAnnealingLR_Warmup(optimizer,total_epoch=100,eta_min=1e-4,warmup_start_lr=1e-4,warmup_epochs=5)

lrs =[]
epochs = []
for epoch in range(100):
    epochs.append(epoch)
    optimizer.step()
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    lrs.append(scheduler.get_last_lr())
    scheduler.step()

plt.figure()
plt.plot(epochs,lrs)
plt.show()

