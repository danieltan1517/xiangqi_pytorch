import torch
path = 'model'

class NNUE(torch.nn.Module):

    def __init__(self):
        super(NNUE, self).__init__()
        self.feature = torch.nn.Linear(7290, 256)
        self.output  = torch.nn.Linear(512, 1)

    def forward(self, white, black):
        white = self.feature(white)
        black = self.feature(black)
        accum = torch.clamp(torch.cat([white, black], dim=1), 0.0, 1.0)
        return torch.sigmoid(self.output(accum))



model = torch.load(path)
print(model.state_dict())
