import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, Dot, Activation, Concatenate
from models.luong.attention import LuongAttention


class Decoder(tf.keras.Model):
    def __init__(self, outputVocabSize, embeddingDims, lstmUnits, outputLength):
        super(Decoder, self).__init__()

        # self.embedding_dim = embedding_dim
        # self.dec_units = dec_units

        self.embedding = Embedding(
            outputVocabSize,
            embeddingDims,
            input_length=outputLength,
            mask_zero=True,
            name='decoderEmbedding'
        )

        self.lstm = LSTM(
            lstmUnits,
            return_sequences=True,
            name="decoderLSMT"
        )
        
        self.attention = LuongAttention()

        self.concat = Concatenate(name="combinedContext")

    def call(self, decoderEmbeddingInput, encoderLSTMOutput, encoderLastState):
         # Convert input to embeddings
        decoderEmbeddingOutput = self.embedding(decoderEmbeddingInput)

        # Run in trough the LSTM
        decoderLSTMOutput = self.lstm(decoderEmbeddingOutput, initial_state=[encoderLastState, encoderLastState])
        
        context_vector, attention_weights = self.attention(decoderLSTMOutput, encoderLSTMOutput)
        
        decoderCombinedContext = self.concat([context_vector, decoderLSTMOutput])

        self.attention_weights = attention_weights

        return decoderCombinedContext, attention_weights