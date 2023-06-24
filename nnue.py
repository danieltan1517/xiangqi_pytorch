import tensorflow
import numpy
import re

#### Copy from Tensorflow NNUE:
#https://github.com/DanielUranga/TensorFlowNNUE/blob/master/train/train.py


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

    def flat(inp):
        inp = inp.flatten()
        return tensorflow.sparse.from_dense(inp)
        
    input1 = flat(input1)
    input2 = flat(input2)
    input3 = flat(input3)
    input4 = flat(input4)
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
    output_layer  = tensorflow.keras.layers.Dense(1, name='output_layer', use_bias=True)

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

def get_lines(filename):
    with open(filename) as file:
        return file.readlines()
    return None

def generate_xiangqi_data(filename):
    # TODO: find a good scale factor.
    SCALE_FACTOR = 310
    while True:
        lines = get_lines(filename)
        for line in lines:
            tokens = line.split(',')
            evaluation = int(tokens[0])
            if abs(evaluation) > 600:
                continue
            evaluation = tensorflow.math.sigmoid(evaluation / SCALE_FACTOR)
            fen_string = tokens[1].rstrip().lstrip()[1:-2]
            x0, x1, x2, x3 = parse_fen_to_indices(fen_string)
            yield (x0, x1, x2, x3), float(evaluation)


train_dataset = tensorflow.data.Dataset.from_generator (
    generate_xiangqi_data, 
    args = ['xiangqi_evaluations.txt'],
    output_signature = (
        (tensorflow.SparseTensorSpec(shape=(7290,), dtype=tensorflow.bool), 
         tensorflow.SparseTensorSpec(shape=(7290,), dtype=tensorflow.bool), 
         tensorflow.SparseTensorSpec(shape=(7290,), dtype=tensorflow.bool), 
         tensorflow.SparseTensorSpec(shape=(7290,), dtype=tensorflow.bool)), 
        tensorflow.TensorSpec(shape=(), dtype=tensorflow.float32)
    )
)

batch_size = 1024
checkpoint_path = "model/cp-{epoch:04d}.ckpt"
cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_path, 
    verbose = 1, 
    save_weights_only = True,
    save_freq = batch_size 
)


model = create_nnue_model()
optimizer = tensorflow.keras.optimizers.AdamW(
    learning_rate = 0.00001
)

model.compile(
    optimizer = optimizer,
    loss = 'mse',
    metrics=[
        tensorflow.keras.metrics.MeanSquaredError(),
    ]
)

model.fit (
    train_dataset.batch(batch_size),
    steps_per_epoch = batch_size / 2,
    callbacks = [cp_callback],
    epochs = 64,
    validation_steps = 128,
)

