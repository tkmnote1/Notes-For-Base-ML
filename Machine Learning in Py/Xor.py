import torch
import torch.nn as nn

X=torch.tensor([
    [0.,0.],
    [0.,1.],
    [1.,0.],
    [1.,1.]
])
y=torch.tensor([
    [0.],
    [1.],
    [1.],
    [0.]
])
# Xor = and (Or + NAND)
class Xormodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(2,2), # 2 input -> 2 hidden neurons
            nn.Sigmoid(), 
            nn.Linear(2,1), # 2 hidden -> 1 output
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.net(x)
model = Xormodel()

crit=nn.MSELoss()
opti = torch.optim.SGD(model.parameters(), lr=1.0)

## training
epochs=10000
for epoch in range(epochs):
    # forward
    y_pred = model(X)
    #loss
    loss=crit(y_pred,y)
    #zero_grad
    opti.zero_grad()
    # backprop
    loss.backward()
    # update weight
    opti.step()
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.6f}")

# Test

print("\nPrediction:")
with torch.no_grad():
    y_pred = model(X)
    print(y_pred)

    y_class = (y_pred >= 0.5).float()
    print("\nClass:")
    print(y_class)


