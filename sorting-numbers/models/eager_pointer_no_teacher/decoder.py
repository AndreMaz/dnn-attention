import tensorflow as tf
from models.eager_pointer_no_teacher.pointer_attention import PointerAttention
from tensorflow.keras.layers import Embedding, LSTM, Layer


class Decoder(Layer):
    def __init__(self, SOS_CODE, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()

        self.SOS_CODE = SOS_CODE
        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.vocab_size = vocab_size

        self.embedding = Embedding(vocab_size, embedding_dim)

        # self.lstm = LSTM(self.dec_units, return_sequences=True)

        # Attention Layers
        self.attention = PointerAttention(self.dec_units, self.vocab_size)
        
        # We are going to do the looping manually so instead of LSMT Layer we use LSTM cell
        self.cell = tf.keras.layers.LSTMCell(
            self.dec_units, recurrent_initializer='glorot_uniform')


    def call(self, encoder_input, dec_hidden, enc_outputs):
        
        # Decoder's input starts with SOS code
        sos_tensor = tf.fill([enc_outputs.shape[0], 1], self.SOS_CODE)
        # Remaining are set to zero
        zeroed_tensor = tf.zeros([enc_outputs.shape[0], enc_outputs.shape[1] - 1], dtype="int32")
        
        # Create the actual decoder's input
        # shape = [batch_size, sequence_size]
        dec_input = tf.concat([sos_tensor, zeroed_tensor], 1)

        # Convert input to embeddings
        dec_input = self.embedding(dec_input)

        prevDecoderHiddenState = dec_hidden[0]
        prevDecoderCarryState = dec_hidden[1]

        perStepInputs = tf.unstack(dec_input, axis=1)
        perStepOutputs = []

        for _, currentInput in enumerate(perStepInputs):
            # Pass the data into LSTM cell
            stepOutput, currentState = self.cell(
                currentInput, states=[prevDecoderHiddenState, prevDecoderCarryState])

            # Update prev states. They will be used in the next iteration
            prevDecoderHiddenState = currentState[0]
            prevDecoderCarryState = currentState[1]

            # Compute the pointers
            pointer = self.attention(stepOutput, enc_outputs)

            # Update decoder's input
            pointed_values = pointer.numpy().argmax(axis=1)

            perStepOutputs.append(pointer)
        
        # return pointers
        return perStepOutputs