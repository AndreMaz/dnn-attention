import time
import tensorflow as tf
import numpy as np
import sys
# For plotting
import matplotlib.pyplot as plt

from utils.read_configs import get_configs
from dataset.generator import generateDataset
from models.transformer_masking.model import Transformer

configs = get_configs(sys.argv)

num_layers = 1
d_model = 64
dff = 64
num_heads = 8

input_vocab_size = configs['vocab_size']
target_vocab_size = configs['vocab_size']
dropout_rate = 0.01

#####################
##### OPTIMIZER #####
#####################

optimizer = tf.keras.optimizers.Adam()

############################
##### LOSS AND METRICS #####
############################

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction='none')
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

###################
##### MASKING #####
###################
def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp)
  
  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp)
  
  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by 
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
  return enc_padding_mask, combined_mask, dec_padding_mask

############################
#### TRANSFORMER MODEL #####
############################

transformer = Transformer(num_layers,
                          d_model,
                          num_heads,
                          dff,
                          input_vocab_size,
                          target_vocab_size, 
                          configs['SOS_CODE'],
                          pe_input=input_vocab_size, 
                          pe_target=target_vocab_size,
                          rate=dropout_rate
                        )

####################
#### TRAIN STEP ####
####################

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
]

# @tf.function(input_signature=train_step_signature)
def train_step(encoder_input, decoder_output):
  #enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
  #  encoder_input, tar_inp
  # )
  
  # Init the decoder's input
  # Decoder's input starts with SOS code
  batch_size = encoder_input.shape[0]
  dec_input = tf.fill([batch_size, 1], configs['SOS_CODE'])


  with tf.GradientTape() as tape:
    combined_attention = transformer(encoder_input,
                                     dec_input,
                                     True,
                                     None,
                                     None,
                                     None
                                     )
    # loss = loss_function(tar_real, predictions)
    loss = loss_object(decoder_output, combined_attention)

  gradients = tape.gradient(loss, transformer.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  
  train_loss(loss)
  train_accuracy(decoder_output, combined_attention)


def train(EPOCHS = 2, batch_size = 128):
    # Generate dataset

    print('Generating dataset...')
    enc_input, dec_input, dec_out = generateDataset(
        configs['num_samples_training'],
        configs['sample_length'],
        configs['min_value'],
        configs['max_value'],
        configs['SOS_CODE'],
        configs['EOS_CODE'],
        configs['vocab_size']
    )
    print('Dataset generated!')
    # test(test_data)

    num_batches = int(len(enc_input) / batch_size)

    for epoch in range(EPOCHS):
      start = time.time()
      
      train_loss.reset_states()
      train_accuracy.reset_states()

      for batch in range(num_batches):
          inp = enc_input[batch * batch_size: (batch+1) * batch_size]
          # tar = dec_input[batch * batch_size: (batch+1) * batch_size]
          real = dec_out[batch * batch_size: (batch+1) * batch_size]

          train_step(inp, real)

          if batch % 50 == 0:
             print ('Epoch {} Batch {} Loss {:.6f} Accuracy {:.6f}'.format(
                 epoch + 1, batch, train_loss.result(), train_accuracy.result()))
            
          # if (epoch + 1) % 5 == 0:
          #   ckpt_save_path = ckpt_manager.save()
          #   print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
          #                                                       ckpt_save_path))
          
      print ('Epoch {} Loss {:.6f} Accuracy {:.6f}'.format(epoch + 1, 
                                                      train_loss.result(), 
                                                      train_accuracy.result()))

      # print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


def tester(model, configs, eager = False, toPlotAttention = False, with_trainer = True):
    num_samples_tests = configs['num_samples_tests']
    correctPredictions = 0
    wrongPredictions = 0

    trainEncoderInput, _, _  = generateDataset(
        configs['num_samples_tests'],
        configs['sample_length'],
        configs['min_value'],
        configs['max_value'],
        configs['SOS_CODE'],
        configs['EOS_CODE'],
        configs['vocab_size'])
    for _, inputEntry in enumerate(trainEncoderInput):
        print('__________________________________________________')

        # print number sequence without EOS
        print(list(inputEntry.numpy().astype("int16")[1:]))

        # Generate correct answer
        correctAnswer = list(inputEntry.numpy().astype("int16"))
        # Remove EOS, sort the numbers and print the correct result
        correctAnswer = correctAnswer[1:]
        correctAnswer.sort()
        print(correctAnswer)

        # Add the batch dimension [batch=1, features]
        inputEntry = tf.expand_dims(inputEntry, 0)

        # Run the inference and generate predicted output
        predictedAnswer, attention_weights = evaluate(model,
                                                      inputEntry,
                                                      configs['input_length'],
                                                      configs['SOS_CODE'],
                                                      eager,
                                                      with_trainer
                                                      )
        print(predictedAnswer)

        if (toPlotAttention == True):
            plotAttention(attention_weights, inputEntry)

        # Compute the diff between the correct answer and predicted
        # If diff is equal to 0 then numbers are sorted correctly
        diff = []
        for index, _ in enumerate(correctAnswer):
            diff.append(correctAnswer[index] - predictedAnswer[index])

        # If all numbers are equal to 0
        if (all(result == 0 for (result) in diff)):
            correctPredictions += 1
            print('______________________OK__________________________')
        else:
            wrongPredictions += 1
            print('_____________________WRONG!_______________________')

    print(
        f"Correct Predictions: {correctPredictions/num_samples_tests} || Wrong Predictions: {wrongPredictions/num_samples_tests}")

def evaluate(model, encoder_input, input_length, SOS_CODE, eager = False, with_trainer = True):

    # Init the decoder's input
    # Decoder's input starts with SOS code
    batch_size = encoder_input.shape[0]
    dec_input = tf.fill([batch_size, 1], configs['SOS_CODE'])

    attention_weights = transformer(encoder_input,
                                    dec_input,
                                    False,
                                    None,
                                    None,
                                    None
                                    )
    # encoder_input = np.reshape(encoder_input, (1, time_steps))

    pointer_index = attention_weights.numpy().argmax(-1)[0]
    
    encoder_input = encoder_input.numpy()
    output = []
    for pointer in pointer_index:
      output.append(int(encoder_input[0, pointer]))
    
    output = output[:-1]

    return output, attention_weights

def plotAttention(attention_weights, inputEntry):
    # print(attention_weights[0].shape)
    plt.matshow(attention_weights[0])

    xTicksNames = list(inputEntry[0].numpy().astype("int16"))
    inputLength = len(xTicksNames)

    yTicksNames = []
    for i in range(inputLength):
        yTicksNames.append(f"step {i}")

    plt.yticks(range(inputLength), yTicksNames)

    plt.xticks(range(inputLength), xTicksNames)

    plt.ylabel('Pointer Probability')
    plt.xlabel('Input Sequence')

    plt.show(block=True)

if __name__ == "__main__":
    train()
    tester(transformer, configs, toPlotAttention=True, with_trainer=True)
