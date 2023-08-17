import torch
import time
import random
import numpy
import re

SCALE_FACTOR = 660.0
path = 'model'

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

    input1 = numpy.zeros(shape = (3,9,90), dtype=numpy.int32)
    input2 = numpy.zeros(shape = (3,9,90), dtype=numpy.int32)

    def mirror_values(stm: bool, ksq: int, piece: int, sq: int, mirror):
        ksq = king_sq_index[mirror[ksq]]
        piece = get_piece_type(piece, stm)
        sq = mirror[sq]
        return (ksq, piece, sq)

    for piece_data in pieces:
        (piece, sq) = piece_data
        ksq, piece, sq = mirror_values(True, red_king_sq, piece, sq, identity)
        input1[ksq][piece][sq] = 1 
        ksq, piece, sq = mirror_values(False, blk_king_sq, piece, sq, mirror_id)
        input2[ksq][piece][sq] = 1 


    def flat(inp):
        inp = inp.flatten()
        return inp
        #return torch.from_numpy(inp)
        
    input1 = flat(input1)
    input2 = flat(input2)
    if stm == True:
        return (input1, input2)
    else:
        return (input2, input1)

def sigmoid(z):
    return (1.0 / (1.0 + numpy.exp(-z))).astype(numpy.float32)

class NNUE(torch.nn.Module):

    def __init__(self):
        super(NNUE, self).__init__()
        self.feature = torch.nn.Linear(2430, 256)
        self.output  = torch.nn.Linear(512, 1)

    def forward(self, white, black):
        white = self.feature(white)
        black = self.feature(black)
        accum = torch.clamp(torch.cat([white, black], dim=1), 0.0, 1.0)
        return torch.sigmoid(self.output(accum))

model = torch.load(path, map_location=torch.device('cpu'))
params = model.state_dict()

# get the weights/biases of the model.
feature_weight = params['feature.weight']
feature_biases = params['feature.bias']
output_weight  = params['output.weight']
output_biases  = params['output.bias']

# simple quantization scheme: scale up the weights/biases. multiple by 127 and round.
feature_weight = torch.round(feature_weight * 127.0)
feature_biases = torch.round(feature_biases * 127.0)
output_weight  = torch.round(output_weight  * 127.0)
output_biases  = torch.round(output_biases  * 127.0 * 127.0)

# convert to numpy.
feature_weight = feature_weight.numpy().astype(numpy.int16)
feature_biases = feature_biases.numpy().astype(numpy.int16)
output_weight  = output_weight.numpy().astype(numpy.int8)
output_biases  = output_biases.numpy().astype(numpy.int32)

def evaluation(actual, fen):
  W,B = parse_fen_to_indices(fen)
  W = feature_weight @ W + feature_biases
  B = feature_weight @ B + feature_biases
  accum = numpy.clip(numpy.concatenate([W,B]), 0, 127)
  result = (output_weight @ accum + output_biases)[0]
  result //= 16
  print(f'Guess: {result}, actual: {actual}')

evaluation(-107, "2rakabr1/9/4b1n2/5R2p/p1c1p1p2/1R6P/P1P3c2/N3C1N2/4A4/2B1KAB2 w - - 0 1")
evaluation(369, "2b1ka3/4a4/2R1b4/3r5/4R4/6P2/c8/9/4A4/2BAK1B2 w - - 0 1")
evaluation(-251, "3ak2C1/9/2n1bR2b/8p/9/2P1r3P/9/4B4/4A4/2BA1K3 b - - 0 1")
evaluation(65, "r3kab2/4a4/2n1b4/p1p1p3p/9/2P1P1B2/PR6R/C1N1C4/9/2BcKA1r1 b - - 0 1")

# create binary data:

with open('xiangqi.nnue', 'wb') as f:

    # write the feature weights.
    feature_weight = feature_weight.transpose()
    print('feature weight shape:',feature_weight.shape)
    f.write(feature_weight.tobytes())

    # write the feature biases.
    f.write(feature_biases.tobytes())

    # write the output weights.
    f.write(output_weight.tobytes())

    # write the output biases
    f.write(output_biases.tobytes())


