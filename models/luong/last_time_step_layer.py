import tensorflow

class GetLastTimestepLayer(Layer):
    def __init__(self, num_outputs):
        super(GetLastTimestepLayer, self).__init__()
        self.num_outputs = num_outputs
    
    def call(self, input):
        # inputRank = len(input.shape)
        return input.gather([input.shape[1]-1], 1).squeeze([1])