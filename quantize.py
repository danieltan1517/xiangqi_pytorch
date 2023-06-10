import tensorflow

def create_nnue_model():
    ''' 
    create nnue model:
    9 = number of king squares
    9 = number of pieces which are not a king
    90 = number of squares
    9x9x90 = 7290

    reflect the board vertically mirror features x2
    7290x2 = 14580
    '''
    input1 = tensorflow.keras.Input(shape=(14580,), sparse=True)
    input2 = tensorflow.keras.Input(shape=(14580,), sparse=True)
    feature_layer = tensorflow.keras.layers.Dense(32, name='feature_layer')
    feature_relu  = tensorflow.keras.layers.ReLU(max_value = 1.0)
    concatenate   = tensorflow.keras.layers.Concatenate()
    output_layer  = tensorflow.keras.layers.Dense(1, name='output_layer')

    # create model
    transform1    = feature_relu(feature_layer(input1))
    transform2    = feature_relu(feature_layer(input2))
    transform     = concatenate([transform1, transform2])
    outputs       = output_layer(transform)
    model         = tensorflow.keras.Model(
        inputs = (input1, input2), 
        outputs = outputs,
        name = 'SHalfKP',
    )
    return model 

latest = tensorflow.train.latest_checkpoint('/home/danieltan/Documents/xiangqi_tensorflow/model')

model = create_nnue_model()
model.load_weights(latest).expect_partial()

layer1_weights  = model.variables[0]
layer1_biases   = model.variables[1]
layer2_weights  = model.variables[2]
layer2_biases   = model.variables[3]

(layer1_weights,_,_)  = tensorflow.quantization.quantize(layer1_weights, -127, 127, tensorflow.qint16)
(layer1_biases,_,_)   = tensorflow.quantization.quantize(layer1_biases,  -127, 127, tensorflow.qint16)
(layer2_weights,_,_)  = tensorflow.quantization.quantize(layer2_weights, -127, 127, tensorflow.qint16)
(layer2_biases,_,_)   = tensorflow.quantization.quantize(layer2_biases,  -127, 127, tensorflow.qint16)


layer1_weights = layer1_weights.numpy().transpose()
layer1_biases = layer1_biases.numpy()
layer2_weights = layer2_weights.numpy()
layer2_biases = layer2_biases.numpy()

