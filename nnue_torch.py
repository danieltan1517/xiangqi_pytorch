import pandas
import torch
import time
import random
import numpy
import re

epochs = 32
batch_size = 262144
learning_rate = 8.75e-4
device = "cuda"  # either 'cpu' or 'cuda'
path = "model" # path of saved model
filename = "xiangqi_evaluations.txt"


identity = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,
             9, 10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 26,
            27, 28, 29, 30, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 44, 
            45, 46, 47, 48, 49, 50, 51, 52, 53, 
            54, 55, 56, 57, 58, 59, 60, 61, 62,
            63, 64, 65, 66, 67, 68, 69, 70, 71,
            72, 73, 74, 75, 76, 77, 78, 79, 80,
            81, 82, 83, 84, 85, 86, 87, 88, 89]

mirror_id = [81, 82, 83, 84, 85, 86, 87, 88, 89,
             72, 73, 74, 75, 76, 77, 78, 79, 80,
             63, 64, 65, 66, 67, 68, 69, 70, 71,
             54, 55, 56, 57, 58, 59, 60, 61, 62,
             45, 46, 47, 48, 49, 50, 51, 52, 53, 
             36, 37, 38, 39, 40, 41, 42, 43, 44, 
             27, 28, 29, 30, 31, 32, 33, 34, 35,
             18, 19, 20, 21, 22, 23, 24, 25, 26,
              9, 10, 11, 12, 13, 14, 15, 16, 17,
              0,  1,  2,  3,  4,  5,  6,  7,  8]


king_sq_index = [None, None, None,  1,  0,  1, None, None, None,
                 None, None, None,  2,  1,  2, None, None, None,
                 None, None, None,  2,  2,  2, None, None, None]

def get_piece(p):
  if p == 'a' or p == 'A' or p == 'B' or p == 'b':
    return 0
  elif p == 'N':
    return 1
  elif p == 'R':
    return 2
  elif p == 'C':
    return 3
  elif p == 'P':
    return 4
  elif p == 'n':
    return 5
  elif p == 'r':
    return 6
  elif p == 'c':
    return 7
  elif p == 'p':
    return 8
  elif p == 'K':
    return 9
  elif p == 'k':
    return 10
  print(p)
  assert(False)

# stm = side to move. True=red. False=black.
def get_piece_type(p, stm):
    if p == 0:
        return 0
    if stm == True:
        return p
    if p >= 5:
        return p - 4
    else:
        return p + 4

def parse_fen_to_indices(fen):
    fen = fen.strip()
    fen = fen[1:-1]
    tokens = re.split('\s+', fen)
    lines  = tokens[0].split('/')
    red_king_sq = None
    blk_king_sq = None
    row = 81
    row_number = 0
    pieces = []
    while row >= 0:
        index = row
        for p in lines[row_number]:
            if p == 'K':
                red_king_sq = index
                index += 1
                continue
            if p == 'k':
                blk_king_sq = index
                index += 1
                continue
            if p >= '1' and p <= '9':
                num = ord(p) - ord('0')
                index += num
            else:
                piece_type = get_piece(p)
                pieces.append([piece_type, index])
                index += 1
        row -= 9
        row_number += 1
    
    stm = False
    if tokens[1] == 'w':
        stm = True
    elif tokens[1] == 'b':
        stm = False

    input1 = numpy.zeros(shape = (3,9,90), dtype=numpy.float32)
    input2 = numpy.zeros(shape = (3,9,90), dtype=numpy.float32)

    def mirror_values(stm: bool, ksq: int, piece: int, sq: int, mirror):
        ksq = king_sq_index[mirror[ksq]]
        piece = get_piece_type(piece, stm)
        sq = mirror[sq]
        return (ksq, piece, sq)

    for piece_data in pieces:
        (piece, sq) = piece_data
        ksq, piece, sq = mirror_values(True, red_king_sq, piece, sq, identity)
        input1[ksq][piece][sq] = 1.0 
        ksq, piece, sq = mirror_values(False, blk_king_sq, piece, sq, mirror_id)
        input2[ksq][piece][sq] = 1.0 


    def flat(inp):
        inp = inp.flatten()
        return torch.from_numpy(inp)
        
    input1 = flat(input1)
    input2 = flat(input2)
    if stm == True:
        return (input1, input2)
    else:
        return (input2, input1)

def sigmoid(z):
    return (1.0 / (1.0 + numpy.exp(-z))).astype(numpy.float32)

class XiangqiDataset(torch.utils.data.Dataset):

    def __init__(self, filename):
        super(XiangqiDataset, self).__init__()
        dataframe = pandas.read_csv(filename, sep=", ", dtype={'eval':numpy.int16, 'positions':str})
        self.evals = dataframe['eval']
        self.positions = dataframe['positions']
        self.length = len(self.evals)

    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        evaluation = self.evals[idx]
        evaluation = torch.tensor([sigmoid(evaluation / 400.0)])
        fen = self.positions[idx]
        white, black = parse_fen_to_indices(fen)
        return (white,black), evaluation

class NNUE(torch.nn.Module):

    def __init__(self):
        super(NNUE, self).__init__()
        self.feature = torch.nn.Linear(2430, 16)
        self.output  = torch.nn.Linear(32, 1)

    def forward(self, white, black):
        white = self.feature(white)
        black = self.feature(black)
        accum = torch.clamp(torch.cat([white, black], dim=1), 0.0, 1.0)
        return torch.sigmoid(self.output(accum))


n = NNUE()
model     = n.to(torch.device(device))
mse_error = torch.nn.MSELoss()
opt       = torch.optim.Adam(model.parameters(), lr=learning_rate)
dataset = XiangqiDataset(filename)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
print('Xiangqi NNUE Data Loaded Successfully.')


for e in range(epochs):
  print(f'Starting Epoch {e+1:3} out of {epochs:4}...')
  start = time.time()
  for data in dataloader:
    ((white,black), evaluation) = data
    # forward propagation.
    white = white.to(device=device)
    black = black.to(device=device)
    evaluation = evaluation.to(device=device)
    score = model(white, black)
    # back propagation.
    loss = mse_error(score, evaluation)
    opt.zero_grad()
    loss.backward()
    opt.step()

  end = time.time()
  time_taken = end - start
  print(f'Finished Epoch {e+1:3} out of {epochs:4}. Mean Squared Error Loss: {loss:8.4e}. Time Taken: {time_taken:8.4e} seconds')
  # backup in case the computer crashes...
  torch.save(model, path)


torch.save(model, path)
