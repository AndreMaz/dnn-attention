import tensorflow as tf
from models.pointer_network.pointer_attention import PointerAttention
from models.pointer_network.pointer_attention import PointerAttentionNoTrainer

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.vocab_size = vocab_size

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)

        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                       return_sequences=True)

        # Attention Layers
        self.attention = PointerAttention(self.dec_units, self.vocab_size)
        
    def call(self, dec_input, dec_hidden, enc_outputs):

        # Convert input to embeddings
        dec_input = self.embedding(dec_input)

        # Pass through LSTM
        decoder_outputs = self.lstm(dec_input, initial_state = dec_hidden )

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

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)

        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                       return_sequences=True)

        # We are going to do the looping manually so instead of LSMT Layer we use LSTM cell
        self.cell = tf.keras.layers.LSTMCell(
            self.dec_units, recurrent_initializer='glorot_uniform')

        # Attention Layers
        self.attention = PointerAttentionNoTrainer(self.dec_units, self.vocab_size)
        
    def call(self, dec_hidden, enc_outputs, enc_input):
        
        # Decoder's input starts with SOS code	
        sos_tensor = tf.fill([enc_outputs.shape[0], 1], self.SOS_CODE)	
        # Remaining are set to zero	
        zeroed_tensor = tf.zeros([enc_outputs.shape[0], enc_outputs.shape[1] - 1], dtype="int32")

        # Create the actual decoder's input	
        # shape = [batch_size, sequence_size]	
        dec_input = tf.concat([sos_tensor, zeroed_tensor], 1)
        dec_input = dec_input.numpy()

        prevDecoderHiddenState = dec_hidden[0]
        prevDecoderCarryState = dec_hidden[1]

        perStepOutputs = []

        for i in range(1, self.input_length):
            # Convert input to embeddings
            currentInput = self.embedding(dec_input[:, i - 1])
            # print(dec_input[:, i - 1])

            stepOutput, currentState = self.cell(
                currentInput, states=[prevDecoderHiddenState, prevDecoderCarryState])

            # Update prev states. They will be used in the next iteration
            prevDecoderHiddenState = currentState[0]
            prevDecoderCarryState = currentState[1]

            pointers = self.attention(stepOutput, enc_outputs)

            point_index = pointers.numpy().argmax(1)
            
            dec_input[:, i] = point_index

            perStepOutputs.append(pointers)
            
            # print(dec_input)

        ##### LAST PREDICTION ####
        currentInput = self.embedding(dec_input[:, self.input_length - 1])

        stepOutput, currentState = self.cell(
                currentInput, states=[prevDecoderHiddenState, prevDecoderCarryState])

        pointers = self.attention(stepOutput, enc_outputs)

        perStepOutputs.append(pointers)

        ######################################################
        perStepOutputs = tf.convert_to_tensor(perStepOutputs)

        # print(dec_input)
        return tf.transpose(perStepOutputs, [1, 0, 2])