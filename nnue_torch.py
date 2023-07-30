import torch

epochs = 2000

class Network(torch.nn.Module):

  def __init__(self):
    super(Network, self).__init__()
    self.feature = torch.nn.Linear(7290, 256)
    self.output  = torch.nn.Linear(512, 1)

  def forward(self, white, black):
    white = self.feature(white)
    black = self.feature(black)
    accum = torch.clamp(torch.cat([white, black]), 0.0, 1.0)
    return torch.sigmoid(self.output(accum))


n = Network()
model     = n.to(torch.device("cpu"))
mse_error = torch.nn.MSELoss()
opt       = torch.optim.Adam(model.parameters(), lr=0.001)

X = torch.tensor([0.0] * 7290)
Y = torch.tensor([0.0])

for e in range(epochs):

  # forward propagation.
  scores = model(X, X)

  # back propagation.
  loss = mse_error(scores, Y)
  print(f'Epoch: {e+1} out of {epochs}. Loss: {loss}')
  opt.zero_grad()
  loss.backward()
  opt.step()




