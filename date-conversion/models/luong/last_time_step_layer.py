from tensorflow.keras.layers import Layer
from tensorflow import gather, squeeze

# Returns the last time step from Encoder LSTM hidden states
class GetLastTimestepLayer(Layer):
    def __init__(self, **kwargs):
        super(GetLastTimestepLayer, self).__init__(**kwargs)

        
    def call(self, input):
        return squeeze(gather(input, [input.shape[1]-1], axis = 1), axis=[1])