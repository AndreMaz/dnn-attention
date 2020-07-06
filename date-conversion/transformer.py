import time
import tensorflow as tf
import sys

from dataset.date_format import INPUT_VOCAB, OUTPUT_VOCAB, INPUT_LENGTH, OUTPUT_LENGTH, INPUT_FNS, dateTupleToYYYYDashMMDashDD, decode_tensor
from dataset.generator import generateDataSet
from models.transformer.transformer import Transformer
from utils.read_configs import get_configs


num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = len(INPUT_VOCAB) + 2
target_vocab_size = len(OUTPUT_VOCAB) + 2
dropout_rate = 0.1

EPOCHS = 20


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

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          pe_input=input_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)


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


# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


# @tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)

def main():
    configs = get_configs(sys.argv)

    # Generate dataset
    trainEncoderInput, \
        trainDecoderInput, \
        trainDecoderOutput, \
        valEncoderInput, \
        valDecoderInput, \
        valDecoderOutput,\
        testDateTuples = generateDataSet(
            configs['min_year'], configs['max_year'])


    batch_size = 1
    num_batches = int(trainEncoderInput.shape[0] / batch_size)

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        # for (batch, (inp, tar)) in enumerate(2):
        for i in range(num_batches - 1):
          enc_in_batch = trainEncoderInput[i * batch_size: (i+1) * batch_size]
          dec_in_batch = trainDecoderInput[i * batch_size: (i+1) * batch_size]

          train_step(enc_in_batch, dec_in_batch)

        # if batch % 50 == 0:
        # print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        # if (epoch + 1) % 5 == 0:
        #  ckpt_save_path = ckpt_manager.save()
        #  print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            train_loss.result(),
                                                            train_accuracy.result()))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


if __name__ == "__main__":
    main()
