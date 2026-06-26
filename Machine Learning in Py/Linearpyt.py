import torch
import torch.nn as nn

# data
X=torch.tensor([
    [0.],
    [1.],
    [2.],
    [3.],
    [4.],
    [5.],
])
y=torch.tensor([
    [1.],
    [3.],
    [5.],
    [7.],
    [9.],
    [11.]
])

##
class lin_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(1,1),
        )
    def forward(self,x):
        return self.net(x)
##
model=lin_1()
crit=nn.MSELoss()
opti=torch.optim.SGD(model.parameters(),lr=0.01)

epochs=100000

for epoch in range(epochs):
    ## forward
    y_pred=model(X)
    ## loss
    loss=crit(y_pred,y)
    ## zero_grad
    opti.zero_grad()
    ## backward
    loss.backward()
    ## update
    opti.step()
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.6f}")
    
print("\nPrediction:")
with torch.no_grad():
    y_pred = model(X)
    print(y_pred)

print("\nParameters:")
for name, param in model.named_parameters():
    print(name, param.data)

