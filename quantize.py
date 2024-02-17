import torch
import time
import random
import numpy
import re

# IMPORTANT: SCALE_FACTOR between quantize.py and nnue_torch.py MUST MATCH
SCALE_FACTOR = 120.0
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

    input1 = numpy.zeros(shape = (2,9,90), dtype=numpy.int32)
    input2 = numpy.zeros(shape = (2,9,90), dtype=numpy.int32)

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
        self.feature = torch.nn.Linear(1620, 128)
        self.output  = torch.nn.Linear(256, 1)


    def forward(self, white, black):
        white = self.feature(white)
        black = self.feature(black)
        accum = torch.clamp(torch.cat([white, black]), 0.0, 1.0)
        return torch.sigmoid(self.output(accum))


model = torch.load(path, map_location=torch.device('cpu'))
params = model.state_dict()
#TODO: quantize correctly...
# get the weights/biases of the model.
feature_weight = params['feature.weight']
feature_biases = params['feature.bias']
#hidden1_weight = params['hidden1.weight']
#hidden1_biases = params['hidden1.bias']
#hidden2_weight = params['hidden2.weight']
#hidden2_biases = params['hidden2.bias']
output_weight  = params['output.weight']
output_biases  = params['output.bias']


s_a = 127.0
s_w = 64.0
s_o = SCALE_FACTOR

# simple quantization scheme: scale up the weights/biases. multiply by 127 and round.
feature_weight = torch.round(feature_weight * s_a)
feature_biases = torch.round(feature_biases * s_a)

#hidden1_weight = torch.round(hidden1_weight * s_w)
#hidden1_biases = torch.round(hidden1_biases * s_w * s_a)

#hidden2_weight = torch.round(hidden2_weight * s_w)
#hidden2_biases = torch.round(hidden2_biases * s_w * s_a)

output_weight_scaling = 32.0

output_weight  = torch.round(output_weight * output_weight_scaling * s_o / s_a)
output_biases  = torch.round(output_biases * output_weight_scaling * s_o)


def convert_to_int8(tensor):
  return tensor.numpy().astype(numpy.int8)

def convert_to_int16(tensor):
  return tensor.numpy().astype(numpy.int16)

def convert_to_int32(tensor):
  return tensor.numpy().astype(numpy.int32)

# feature
feature_weight = convert_to_int16(feature_weight)
feature_biases = convert_to_int16(feature_biases)

# hidden1
#hidden1_weight = convert_to_int16(hidden1_weight)
#hidden1_biases = convert_to_int16(hidden1_biases)

# hidden2
#hidden2_weight = convert_to_int16(hidden2_weight)
#hidden2_biases = convert_to_int16(hidden2_biases)

# output
output_weight  = convert_to_int8(output_weight)
output_biases  = convert_to_int32(output_biases)

print(output_weight)
print(numpy.min(output_weight))
print(numpy.max(output_weight))
print(output_biases)

def evaluation(actual, fen):
  W,B = parse_fen_to_indices(fen)
  wdl = model(torch.from_numpy(W.astype(numpy.float32)),torch.from_numpy(B.astype(numpy.float32)))
  print(f'FEN: {fen}')
  print(wdl, sigmoid(actual / SCALE_FACTOR))
  W = feature_weight @ W + feature_biases
  B = feature_weight @ B + feature_biases
  accum = numpy.clip(numpy.concatenate([W,B]), 0, 127)
  #accum = numpy.clip((hidden1_weight @ accum + hidden1_biases) // 64, 0, 127)
  #accum = numpy.clip((hidden2_weight @ accum + hidden2_biases) // 64, 0, 127)
  output = (output_weight @ accum + output_biases) // output_weight_scaling
  print(f'Guess: {output}, actual: {actual}')
  print('=======================================')

evaluation(-10,"rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C4NC1/9/RNBAKAB1R b")
evaluation(70,"r2akab2/3r5/n1c1b1c1n/p5p1C/9/C5BR1/P2pP1P1P/N5N2/4A4/R2K1AB2 b")
evaluation(-41,"r2akab2/3r5/n3b1c1n/p5p1C/9/C6R1/P2pP1P1P/N1c1B1N1B/4A4/R2K1A3 w")
evaluation(107,"r2akab2/3r5/n1c1b1c1n/p5p1C/9/C6R1/P2pP1P1P/N3BAN2/9/R2K1AB2 b")


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
