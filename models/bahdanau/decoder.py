import tensorflow as tf
from models.bahdanau.attention import BahdanauAttention

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units):
    super(Decoder, self).__init__()
    # self.batch_sz = batch_sz
    self.embedding_dim = embedding_dim
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

    self.cell = tf.keras.layers.LSTMCell(self.dec_units, recurrent_initializer='glorot_uniform')


  def call(self, dec_input, dec_hidden, enc_outputs):

    ## Convert input to embeddings
    dec_input = self.embedding(dec_input)

    # Reshape from batch-major into time-major
    # [time_steps, batch_size, features]
    dec_input = tf.transpose(dec_input, [1,0,2])
    # tf.reshape(x, [1,0,2])
    perStepInputs = tf.unstack(dec_input, axis=0)
    perStepOutputs = []

    prevDecoderHiddenState = dec_hidden[0]
    prevDecoderCarryState = dec_hidden[1]

    for _, currentInput in enumerate(perStepInputs):
        # print(currentInput.shape)
        
        contextVector, attention_weights = self.attention(prevDecoderHiddenState, enc_outputs)

        currentInput = tf.concat([currentInput, contextVector], axis=1)
        # print(currentInput.shape)

        ## Concatenate with current input
        stepOutput, currentState = self.cell(currentInput, states=[prevDecoderHiddenState, prevDecoderCarryState])
        prevDecoderHiddenState = currentState[0]
        prevDecoderCarryState = currentState[1]
        # print(stepOutput)
        # print(prevDecoderHiddenState)
        perStepOutputs.append(stepOutput)

    # Convert list to tensor
    outTensor = tf.convert_to_tensor(perStepOutputs, dtype="float32")

    # Reshape back into batch-major
    # [batch_size, time_steps, features]
    return tf.transpose(outTensor, [1,0,2])