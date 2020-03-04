import tensorflow as tf
from models.pointer_network.pointer_attention import PointerAttention


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.dec_units = dec_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # Attention Layers
        self.attention = PointerAttention(self.dec_units)

        # We are going to do the looping manually so instead of LSMT Layer we use LSTM cell
        self.cell = tf.keras.layers.LSTMCell(
            self.dec_units, recurrent_initializer='glorot_uniform')

        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                       return_sequences=True,
                       return_state=True)


    def call(self, dec_input, dec_hidden, enc_outputs):

        # Convert input to embeddings
        dec_input = self.embedding(dec_input)

        decoder_outputs, hidden, carry = self.lstm(dec_input, initial_state = dec_hidden )

        pointers = self.attention(decoder_outputs, enc_outputs)

        # perStepInputs = tf.unstack(dec_input, axis=1)
        # perStepOutputs = []

        # prevDecoderHiddenState = dec_hidden[0]
        # prevDecoderCarryState = dec_hidden[1]

        # pointerList = []
        # # Iterate over time steps and compute the attention
        # for _, currentInput in enumerate(perStepInputs):
        #     # Pass the data into LSTM cell
        #     stepOutput, currentState = self.cell(
        #         currentInput, states=[prevDecoderHiddenState, prevDecoderCarryState])

        #     # Update prev states. They will be used in the next iteration
        #     prevDecoderHiddenState = currentState[0]
        #     prevDecoderCarryState = currentState[1]

        #     pointer = self.attention(stepOutput, enc_outputs)
        #     pointerList.append(pointerList)


        # decoderHiddenStates, decoderLastHiddenState, decoderLastCarryState = self.lstm(dec_input, initial_state = dec_hidden)

        # pointers = self.attention(decoderHiddenStates, enc_outputs)

        # pointers = tf.convert_to_tensor(pointers)

        return pointers
