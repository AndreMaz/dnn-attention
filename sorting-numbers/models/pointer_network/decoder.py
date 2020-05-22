import tensorflow as tf
from models.pointer_network.pointer_attention import PointerAttention
from models.pointer_network.pointer_attention import PointerAttentionNoTrainer


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.vocab_size = vocab_size

        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, mask_zero=True)

        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                         return_sequences=True)

        # Attention Layers
        self.attention = PointerAttention(self.dec_units, self.vocab_size)

    def call(self, dec_input, dec_hidden, enc_outputs):

        # Convert input to embeddings
        dec_input = self.embedding(dec_input)

        # Pass through LSTM
        decoder_outputs = self.lstm(dec_input, initial_state=dec_hidden)

        # Compute the pointers
        pointers = self.attention(decoder_outputs, enc_outputs)

        return pointers


class DecoderNoTrainer(tf.keras.Model):
    def __init__(self, input_length, vocab_size, embedding_dim, dec_units, SOS_CODE):
        super(DecoderNoTrainer, self).__init__()

        self.input_length = input_length
        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.vocab_size = vocab_size
        self.SOS_CODE = SOS_CODE

        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, mask_zero=True)

        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                         return_sequences=True)

        # We are going to do the looping manually so instead of LSMT Layer we use LSTM cell
        self.cell = tf.keras.layers.LSTMCell(
            self.dec_units, recurrent_initializer='glorot_uniform')

        # Attention Layers
        self.attention = PointerAttentionNoTrainer(
            self.dec_units, self.vocab_size)

    def call(self, dec_hidden, enc_outputs, enc_input):
        enc_input = enc_input.numpy()

        # Create a tensor with the batch indices
        batch_indices = tf.convert_to_tensor(
            list(range(enc_outputs.shape[0])), dtype='int32')

        # Decoder's input starts with SOS code
        sos_tensor = tf.fill([enc_outputs.shape[0], 1], self.SOS_CODE)
        # Remaining are set to zero
        zeroed_tensor = tf.zeros(
            [enc_outputs.shape[0], enc_outputs.shape[1] - 1], dtype="int32")

        # Create the actual decoder's input
        # shape = [batch_size, sequence_size]
        dec_input = tf.concat([sos_tensor, zeroed_tensor], 1)
        dec_input = dec_input.numpy()

        # Initial state for the LSTM cell.
        # These values are the last hidden state of the Encoder
        prevDecoderHiddenState = dec_hidden[0]
        prevDecoderCarryState = dec_hidden[1]

        # We will store pointers here
        perStepOutputs = []

        for i in range(1, self.input_length):
            # Get the previous slice of the decoder's input sequence
            # Convert input to embeddings
            currentInput = self.embedding(dec_input[:, i - 1])

            # Call the cell
            stepOutput, currentState = self.cell(
                currentInput, states=[prevDecoderHiddenState, prevDecoderCarryState])

            # Update prev states. They will be used in the next iteration
            prevDecoderHiddenState = currentState[0]
            prevDecoderCarryState = currentState[1]

            # Compute the pointers
            pointers = self.attention(stepOutput, enc_outputs)

            # Grab the indices of the values pointed by the
            point_index = pointers.numpy().argmax(1)

            # Get the value that pointer points to
            # Update the decoder
            dec_input[:, i] = enc_input[batch_indices, point_index]

            # Store pointers
            perStepOutputs.append(pointers)

        # Last prediction step
        # Final pointer given the full decoder's sequence as input
        # If model is trained well it should point to EOS symbol
        currentInput = self.embedding(dec_input[:, self.input_length - 1])

        stepOutput, currentState = self.cell(
            currentInput, states=[prevDecoderHiddenState, prevDecoderCarryState])

        pointers = self.attention(stepOutput, enc_outputs)

        # Store the pointer
        perStepOutputs.append(pointers)

        # Reshape the data back into [batch_size, time_step, pointer]
        perStepOutputs = tf.convert_to_tensor(perStepOutputs)
        return tf.transpose(perStepOutputs, [1, 0, 2])
