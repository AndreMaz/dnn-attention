import time
import tensorflow as tf
import sys
# For plotting
import matplotlib.pyplot as plt

from dataset.date_format import START_CODE, INPUT_VOCAB, OUTPUT_VOCAB, INPUT_LENGTH, OUTPUT_LENGTH, INPUT_FNS, dateTupleToYYYYDashMMDashDD, encodeInputDateStrings, decode_tensor
from dataset.generator import generateDataSet
from models.transformer.model import Transformer

num_layers = 2
d_model = 64
dff = 64
num_heads = 4

input_vocab_size = len(INPUT_VOCAB)
target_vocab_size = len(OUTPUT_VOCAB)
dropout_rate = 0.01

#####################
##### OPTIMIZER #####
#####################
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
optimizer = tf.keras.optimizers.Adam()

############################
##### LOSS AND METRICS #####
############################
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  
  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


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

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, 
                          pe_input=input_vocab_size, 
                          pe_target=target_vocab_size,
                          rate=dropout_rate)
######################
#### CHECKPOINTS #####
######################
# checkpoint_path = "./checkpoints/train"
# ckpt = tf.train.Checkpoint(transformer=transformer,optimizer=optimizer)

# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
# if ckpt_manager.latest_checkpoint:
  # ckpt.restore(ckpt_manager.latest_checkpoint)
  # print ('Latest checkpoint restored!!')

####################
#### TRAIN STEP ####
####################

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar, real):
  tar_inp = tar # tar[:, :-1]
  tar_real = real # tar[:, 1:]
  
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  
  with tf.GradientTape() as tape:
    predictions, _ = transformer(inp, tar_inp, 
                                 True, 
                                 enc_padding_mask, 
                                 combined_mask, 
                                 dec_padding_mask)
    # loss = loss_function(tar_real, predictions)
    loss = loss_object(tar_real, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  
  train_loss(loss)
  train_accuracy(tar_real, predictions)


def train(min_year="1950-01-01", max_year="2050-01-01", EPOCHS = 30, batch_size = 512):
    # Generate dataset

    print('Generating dataset...')
    enc_input, dec_input, dec_out, _, _, _, test_data = generateDataSet(
        min_year, max_year, dec_output_one_hot=False, trainSplit=0.99, validationSplit=0)
    print('Dataset generated!')
    # test(test_data)

    num_batches = int(len(enc_input) / batch_size)

    for epoch in range(EPOCHS):
      start = time.time()
      
      train_loss.reset_states()
      train_accuracy.reset_states()

      for batch in range(num_batches):
          inp = enc_input[batch * batch_size: (batch+1) * batch_size]
          tar = dec_input[batch * batch_size: (batch+1) * batch_size]
          real = dec_out[batch * batch_size: (batch+1) * batch_size]

          train_step(inp, tar, real)

          if batch % 50 == 0:
             print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                 epoch + 1, batch, train_loss.result(), train_accuracy.result()))
            
          # if (epoch + 1) % 5 == 0:
          #   ckpt_save_path = ckpt_manager.save()
          #   print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
          #                                                       ckpt_save_path))
          
      print ('Epoch {} Loss {:.6f} Accuracy {:.6f}'.format(epoch + 1, 
                                                      train_loss.result(), 
                                                      train_accuracy.result()))

      # print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    test(test_data)

####################
#### EVALUATION ####
####################
def test(test_data, num_tests = 10):
  # Do only one test
  # input_str = INPUT_FNS[0](test_data[0])
  # correct_answer = dateTupleToYYYYDashMMDashDD(test_data[0])
  # predicted_answer = convert_date(input_str, correct_answer, plot='decoder_layer4_block2')

  totalTests = len(test_data)*len(INPUT_FNS)
  correctPredictions = 0
  wrongPredictions = 0

  for date_tuple in test_data:
    for _, fn in enumerate(INPUT_FNS):
      print('_________________________________')
      input_str = fn(date_tuple)

      correct_answer = dateTupleToYYYYDashMMDashDD(date_tuple)

      predicted_answer = convert_date(input_str, correct_answer, plot='')

      print('Input: {}'.format(input_str))
      print('Correct   translation: {}'.format(correct_answer))
      print('Predicted translation: {}'.format(predicted_answer))

      if (correct_answer == predicted_answer):
        correctPredictions += 1
        print('CORRECT')
      else:
        wrongPredictions += 1
        print('WRONG!')

      print('---------------------------------')

  print(f"Test Sample Size: {len(test_data)} || Number of Formats {len(INPUT_FNS)} || Total Tests {totalTests}")
  print(f"Correct Predictions: {correctPredictions/totalTests} || Wrong Predictions: {wrongPredictions/totalTests}")

def evaluate(inp_sentence):
  # start_token = [tokenizer_pt.vocab_size]
  # end_token = [tokenizer_pt.vocab_size + 1]
  
  # inp sentence is portuguese, hence adding the start and end token
  # inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
  encoder_input = encodeInputDateStrings([inp_sentence])
  # encoder_input = tf.expand_dims(inp_sentence, 0)
  
  # as the target is english, the first word to the transformer should be the
  # english start token.
  decoder_input = [START_CODE]
  output = tf.expand_dims(decoder_input, 0)
    
  for i in range(OUTPUT_LENGTH):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)
  
    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = transformer(encoder_input, 
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)
    
    # select the last word from the seq_len dimension
    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
    # return the result if the predicted_id is equal to the end token
    # if predicted_id == tokenizer_en.vocab_size+1:
    #  return tf.squeeze(output, axis=0), attention_weights
    
    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0), attention_weights

def plot_attention_weights(attention, sentence, result, layer):
  fig = plt.figure(figsize=(16, 8))
  
  sentence = encodeInputDateStrings([sentence])[0]
  
  # Remove batch dim
  attention = tf.squeeze(attention[layer], axis=0)
  
  for head in range(attention.shape[0]):
    ax = fig.add_subplot(2, 4, head+1)
    
    # plot the attention weights
    ax.matshow(attention[head][:, :], cmap='viridis')

    fontdict = {'fontsize': 10}
    
    ax.set_xticks(range(len(sentence)))
    ax.set_yticks(range(len(result)))
    
    # ax.set_ylim(len(result)-1.5, -0.5)
        
    ax.set_xticklabels([INPUT_VOCAB[int(i)] for i in sentence], 
        fontdict=fontdict, rotation=90)
    
    ax.set_yticklabels([OUTPUT_VOCAB[int(i)] for i in result], 
                       fontdict=fontdict)
    
    ax.set_xlabel('Head {}'.format(head+1))
  
  plt.tight_layout()
  plt.show(block=True)

def convert_date(sentence, correct_answer, plot=''):
  result, attention_weights = evaluate(sentence)
  result = result.numpy()[1:] # Remove SOS symbol

  predicted_sentence = ""
  for i in result:
    predicted_sentence += OUTPUT_VOCAB[int(i)]
   # tokenizer_en.decode([i for i in result])  

  if plot:
    plot_attention_weights(attention_weights, sentence, result, plot)

  return predicted_sentence

if __name__ == "__main__":
    train()
