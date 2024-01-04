import pandas
import torch
import time
import random
import numpy
import re

epochs = 400 
batch_size = 8192
learning_rate = 8.75e-4
device = "cuda"  # either 'cpu' or 'cuda'
path = "model" # path of saved model
filename = "xiangqi_evaluations.txt"
num_workers = 4
SCALE_FACTOR = 360   # IMPORTANT: SCALE_FACTOR between quantize.py and nnue_torch.py MUST MATCH
start_from_scratch = True 

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
                 None, None, None,  1,  1,  1, None, None, None,
                 None, None, None,  1,  1,  1, None, None, None]


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

    input1 = numpy.zeros(shape = (2,9,90), dtype=numpy.float32)
    input2 = numpy.zeros(shape = (2,9,90), dtype=numpy.float32)

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


    def __init__(self, filename: str):
        super(XiangqiDataset, self).__init__()
        dataframe = pandas.read_csv(filename, dtype={'eval':numpy.int16, 'positions':str})
        self.evals = dataframe['eval']
        self.positions = dataframe['positions']
        self.length = len(self.evals)


    def __init__(self, dataset: pandas.DataFrame):
        self.evals = dataset['eval']
        self.positions = dataset['positions']
        self.length = len(self.evals)
        

    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        evaluation = self.evals[idx]
        evaluation = torch.tensor([sigmoid(evaluation / SCALE_FACTOR)])
        fen = self.positions[idx]
        white, black = parse_fen_to_indices(fen)
        return (white,black), evaluation


def create_datasets(filename, factor=0.9, eval_margin=150):
    dataset = pandas.read_csv(filename, dtype={'eval':numpy.int16, 'positions':str})
    dataset = dataset.loc[(dataset['eval'] >= -eval_margin) & (dataset['eval'] <= eval_margin)]
    dataset.reset_index(inplace=True)
    print(f'Loaded {len(dataset)} pairs of data.')
    train = dataset.sample(frac=factor,random_state=0, ignore_index=True)
    test  = dataset.drop(train.index, axis='index')
    test.reset_index(inplace=True)
    train_dataset = XiangqiDataset(train)
    test_dataset = XiangqiDataset(test)
    return train_dataset, test_dataset
    

class NNUE(torch.nn.Module):


    def __init__(self):
        super(NNUE, self).__init__()
        self.feature = torch.nn.Linear(1620, 128)
        self.output  = torch.nn.Linear(256, 1)


    def forward(self, white, black):
        white = self.feature(white)
        black = self.feature(black)
        accum = torch.clamp(torch.cat([white, black], dim=1), 0.0, 1.0)
        return torch.sigmoid(self.output(accum))


def validation_dataset(test_dataloader, model, mse_error):
  with torch.no_grad():
    avg_loss = 0.0
    n = 0
    for data in test_dataloader:
      ((white,black), evaluation) = data
      white = white.to(device=device)
      black = black.to(device=device)
      evaluation = evaluation.to(device=device)
      score = model(white, black)
      loss = mse_error(score, evaluation)
      avg_loss += loss.data.item()
      n += 1
    avg_loss /= n
    return avg_loss

nnue = None
model = None
if start_from_scratch == True:
  nnue = NNUE()
  model = nnue.to(torch.device(device))
else:
  model = torch.load(path, map_location=torch.device(device))

mse_error = torch.nn.MSELoss()
opt       = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_dataset, test_dataset = create_datasets(filename, 0.80, 1500)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_dataloader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
print('Xiangqi NNUE Data Loaded Successfully.')

tolerance = 0
total_loss = float('inf')
centipawn = sigmoid(1 / SCALE_FACTOR) - .5
for e in range(epochs):
  print(f'Starting Epoch {e+1:3} out of {epochs:4}...Centipawn {centipawn}')
  model.train()
  start = time.time()
  for data in train_dataloader:
    ((white,black), evaluation) = data
    # forward propagation.
    white = white.to(device=device)
    black = black.to(device=device)
    evaluation = evaluation.to(device=device)
    score = model(white, black)
    # back propagation.
    opt.zero_grad()
    loss = mse_error(score, evaluation)
    loss.backward()
    opt.step()

  model.eval()
  end = time.time()
  time_taken = end - start
  print(f'Finished Epoch {e+1:3} out of {epochs:4}. Mean Squared Error Loss: {loss:8.4e}. Time Taken: {time_taken:8.4e} seconds')

  start = time.time()
  average_loss = validation_dataset(test_dataloader, model, mse_error)
  end = time.time()
  time_taken = end - start
  print(f'Average Validation Loss: {average_loss}. Total Loss: {total_loss}. Time Taken: {time_taken:8.4e} seconds')
  if average_loss < total_loss:
    # save the model with the lowest average validation loss.
    print('Saving Model...')
    tolerance = 0
    total_loss = average_loss
    torch.save(model, path)
  else:
    tolerance += 1
    if tolerance >= 10:
      print('Model stopped getting better...')
      break





