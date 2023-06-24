import tensorflow
import numpy
import re

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

horizontal_mirror = [81, 82, 83, 84, 85, 86, 87, 88, 89,
                     72, 73, 74, 75, 76, 77, 78, 79, 80,
                     63, 64, 65, 66, 67, 68, 69, 70, 71,
                     54, 55, 56, 57, 58, 59, 60, 61, 62,
                     45, 46, 47, 48, 49, 50, 51, 52, 53, 
                     36, 37, 38, 39, 40, 41, 42, 43, 44, 
                     27, 28, 29, 30, 31, 32, 33, 34, 35,
                     18, 19, 20, 21, 22, 23, 24, 25, 26,
                      9, 10, 11, 12, 13, 14, 15, 16, 17,
                      0,  1,  2,  3,  4,  5,  6,  7,  8]

vertical_mirror = [ 8,  7,  6,  5,  4,  3,  2,  1,  0, 
                   17, 16, 15, 14, 13, 12, 11, 10,  9, 
                   26, 25, 24, 23, 22, 21, 20, 19, 18, 
                   35, 34, 33, 32, 31, 30, 29, 28, 27, 
                   44, 43, 42, 41, 40, 39, 38, 37, 36,  
                   53, 52, 51, 50, 49, 48, 47, 46, 45,  
                   62, 61, 60, 59, 58, 57, 56, 55, 54, 
                   71, 70, 69, 68, 67, 66, 65, 64, 63, 
                   80, 79, 78, 77, 76, 75, 74, 73, 72, 
                   89, 88, 87, 86, 85, 84, 83, 82, 81]

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

# stm = side to move.
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

    input1 = numpy.zeros(shape = (9,9,90), dtype=numpy.bool8)
    input2 = numpy.zeros(shape = (9,9,90), dtype=numpy.bool8)
    input3 = numpy.zeros(shape = (9,9,90), dtype=numpy.bool8)
    input4 = numpy.zeros(shape = (9,9,90), dtype=numpy.bool8)
    
    #TODO: this is wrong. red_king_sq should not be here.
    #it should be relative to turn.

    #TODO: need to validate that this logic makes sense...
    our_king = None
    opp_king = None
    if stm == True:
        our_king = red_king_sq
        opp_king = horizontal_mirror[blk_king_sq]
    else:
        our_king = horizontal_mirror[blk_king_sq]
        opp_king = red_king_sq

    for piece in pieces:
        (piece_type, piece_sq) = piece
        if stm == False:
            piece_sq = horizontal_mirror[piece_sq]

        p = get_piece_type(piece_type, stm)
        input1[king_sq_index[our_king]][p][piece_sq] = True
        input2[king_sq_index[vertical_mirror[our_king]]][p][piece_sq] = True

        p = get_piece_type(piece_type, not stm)
        if stm == False:
            piece_sq = horizontal_mirror[piece_sq]

        input3[king_sq_index[opp_king]][p][piece_sq] = True
        input4[king_sq_index[vertical_mirror[opp_king]]][p][piece_sq] = True
        
    input1 = input1.flatten()
    input2 = input2.flatten()
    input3 = input3.flatten()
    input4 = input4.flatten()
    return (input1, input2, input3, input4)


def create_nnue_model():
    ''' 
    create nnue model:
    9 = number of king squares
    9 = number of pieces which are not a king
    90 = number of squares
    9x9x90 = 7290

    reflect the board vertically mirror features
    '''
    input1 = tensorflow.keras.Input(shape=(7290,), sparse=True)
    input2 = tensorflow.keras.Input(shape=(7290,), sparse=True)
    input3 = tensorflow.keras.Input(shape=(7290,), sparse=True)
    input4 = tensorflow.keras.Input(shape=(7290,), sparse=True)
    feature_layer = tensorflow.keras.layers.Dense(64, name='feature_layer')
    feature_relu  = tensorflow.keras.layers.ReLU(max_value = 1.0)
    concatenate   = tensorflow.keras.layers.Concatenate()
    output_layer  = tensorflow.keras.layers.Dense(1, name='output_layer', use_bias=False)

    # create model
    transform1    = feature_layer(input1)
    transform2    = feature_layer(input2)
    transform3    = feature_layer(input3)
    transform4    = feature_layer(input4)

    transfrom1    = feature_relu(transform1 + transform2)
    transfrom2    = feature_relu(transform3 + transform4)

    transform     = concatenate([transform1, transform2])
    outputs       = output_layer(transform)
    model         = tensorflow.keras.Model(
        inputs = (input1, input2, input3, input4), 
        outputs = outputs,
        name = 'SHalfKP',
    )
    return model 


def quantize(weights, biases, dtype):
    #max_weight = 1 #max(tensorflow.math.reduce_max(weights), tensorflow.math.reduce_max(biases))
    #min_weight = 0 #min(tensorflow.math.reduce_min(weights), tensorflow.math.reduce_min(biases))
    #(weights,_,_) = tensorflow.quantization.quantize(weights, min_weight, max_weight, dtype)
    #(biases,_,_)  = tensorflow.quantization.quantize(biases,  min_weight, max_weight, dtype)
    weights = weights * 2048 * 1.15
    weights = weights.numpy().astype(dtype)
    print(numpy.max(weights))
    print(numpy.min(weights))
    biases  = biases * 2048 * 1.15
    biases  = biases.numpy().astype(dtype).flatten()
    print(numpy.max(biases))
    print(numpy.min(biases))
    print(weights, biases)
    return weights, biases

def quantize_layer2(weights, dtype):
    weights = weights * 2048 * 1.15
    weights = weights.numpy().astype(dtype).transpose()
    print(numpy.max(weights))
    print(numpy.min(weights))
    print(weights)
    return weights

latest = tensorflow.train.latest_checkpoint('/home/danieltan/Documents/xiangqi_tensorflow/model')

model = create_nnue_model()
model.load_weights(latest).expect_partial()

layer1_weights  = model.variables[0]
layer1_biases   = model.variables[1]
layer2_weights  = model.variables[2]

layer1_weights, layer1_biases = quantize(layer1_weights, layer1_biases, numpy.int16)
layer1_weights = layer1_weights.transpose()
layer2_weights = quantize_layer2(layer2_weights, numpy.int8)

# todo: find a good quantization scheme.
def evaluate(fen, expected):
    i1, i2, i3, i4 = parse_fen_to_indices(fen)
    i1 =  layer1_weights.dot(i1) + layer1_biases
    i2 =  layer1_weights.dot(i2) + layer1_biases
    i3 =  layer1_weights.dot(i3) + layer1_biases
    i4 =  layer1_weights.dot(i4) + layer1_biases
    i1 =  numpy.clip((i1 + i2), 0, 0x7F)
    i2 =  numpy.clip((i3 + i4), 0, 0x7F)
    vec = numpy.concatenate([i1,i2])
    vec = layer2_weights.dot(vec)[0] // 128
    print('(quantized, dataset)', (vec, expected), fen)


evaluate("2rakabr1/9/4b1n2/5R2p/p1c1p1p2/1R6P/P1P3c2/N3C1N2/4A4/2B1KAB2 w - - 0 1", -107)
evaluate("4kab2/4a4/4b4/3r5/p3p1c2/5nN2/P1n1P4/2NCBA1R1/2C1K4/3A2Bc1 w - - 0 1", -14)
evaluate("r2akabr1/9/1cn1b4/p1p5p/9/1CP1p1N2/P4R2P/2N1C2c1/7n1/R1BAKAB2 b - - 0 1", 230)
evaluate("2bk1ab2/4a4/6r2/5c3/9/9/7R1/4B4/4A4/2BAK4 b - - 0 1", 283)
evaluate("4k4/4a4/4ba3/6P2/4cNb2/5p3/5n3/3AB3N/4A4/2B2K3 w - - 0 1", 15)

'''
with open('model.nnue', 'wb') as network:
    network.write(layer1_weights.tobytes())
    network.write(layer1_biases.tobytes())
    network.write(layer2_weights.tobytes())
    network.write(layer2_biases.tobytes())
'''

