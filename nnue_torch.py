import torch
import random
import numpy
import re

epochs = 1

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


king_sq_index = [None, None, None,  0,  1,  2, None, None, None,
                 None, None, None,  3,  4,  5, None, None, None,
                 None, None, None,  6,  7,  8, None, None, None]

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

    input1 = numpy.zeros(shape = (9,9,90), dtype=numpy.float32)
    input2 = numpy.zeros(shape = (9,9,90), dtype=numpy.float32)

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

def generate_evaluations(lines, num):
    # TODO: find a good scale factor.
    SCALE_FACTOR = 400
    random.shuffle(lines)
    i = 0
    for line in lines:
        tokens = line.split(',')
        evaluation = int(tokens[0])
        if abs(evaluation) > 600 and abs(evaluation) == 285:
            continue
        evaluation = torch.sigmoid(torch.tensor([evaluation]) / SCALE_FACTOR)
        fen_string = tokens[1].rstrip().lstrip()[1:-2]
        W, B = parse_fen_to_indices(fen_string)
        yield (W,B), evaluation
        i += 1
        if i >= num:
            break

class XiangqiDataset(torch.utils.data.Dataset):

    def __init__(self, filename):
        super(XiangqiDataset, self).__init__()
        lines = None
        self.data = []
        with open(filename) as file:
            pass
            #self.lines = file.readlines()
            #self.lines

    def __len__(self):
        return len(self.lines)


    def __getitem__(self, idx):
        fen = self.lines[idx]
        W,B = parse_fen_to_indices(fen)

class NNUE(torch.nn.Module):

    def __init__(self):
        super(NNUE, self).__init__()
        self.feature = torch.nn.Linear(7290, 256)
        self.output  = torch.nn.Linear(512, 1)

    def forward(self, white, black):
        white = self.feature(white)
        black = self.feature(black)
        accum = torch.clamp(torch.cat([white, black]), 0.0, 1.0)
        return torch.sigmoid(self.output(accum))


n = NNUE()
model     = n.to(torch.device("cuda"))
mse_error = torch.nn.MSELoss()
opt       = torch.optim.Adam(model.parameters(), lr=0.0001)

#W, B = parse_fen_to_indices("2rakabr1/9/4b1n2/5R2p/p1c1p1p2/1R6P/P1P3c2/N3C1N2/4A4/2B1KAB2 w - - 0 1")
#X = torch.tensor([0.0] * 7290)
#Y = torch.tensor([.4])
lines = get_lines('xiangqi_evaluations.txt')

for e in range(epochs):
  print(f'Epoch {e+1} out of {epochs}')
  i = 0
  for data in generate_evaluations(lines, 128):
    (W, B), evaluation = data
    # forward propagation.
    W = W.to(device="cuda")
    B = B.to(device="cuda")
    evaluation = evaluation.to(device="cuda")
    score = model(W, B)

    # back propagation.
    loss = mse_error(score, evaluation)
    print(i, loss)
    i += 1
    opt.zero_grad()
    loss.backward()
    opt.step()




