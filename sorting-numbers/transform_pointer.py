import time
import tensorflow as tf
import sys
# For plotting
import matplotlib.pyplot as plt

from utils.read_configs import get_configs
from dataset.generator import generateDataset
from models.transformer.model import Transformer

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
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  
  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
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
                          pe_input=input_vocab_size, 
                          pe_target=target_vocab_size,
                          rate=dropout_rate
                        )
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
    tf.TensorSpec(shape=(None, None, None), dtype=tf.int64),
]

# @tf.function(input_signature=train_step_signature)
def train_step(inp, tar, real):
  tar_inp = tar # tar[:, :-1]
  tar_real = real # tar[:, 1:]
  
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  
  with tf.GradientTape() as tape:
    combined_attention = transformer(inp,
                                 tar_inp,
                                 True,
                                 enc_padding_mask,
                                 combined_mask,
                                 dec_padding_mask)
    # loss = loss_function(tar_real, predictions)
    loss = loss_object(tar_real, combined_attention)

  gradients = tape.gradient(loss, transformer.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  
  train_loss(loss)
  train_accuracy(tar_real, combined_attention)


def train(EPOCHS = 70, batch_size = 1):
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
          tar = dec_input[batch * batch_size: (batch+1) * batch_size]
          real = dec_out[batch * batch_size: (batch+1) * batch_size]

          train_step(inp, tar, real)

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

if __name__ == "__main__":
    train()
