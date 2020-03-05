# from dataset.dataset_generator import ArtificialDataset
from dataset.generator import generateDataset
from models.model_factory import model_factory

embedding_dims = 64
lstm_units = 64

num_samples = 50_000
sample_length = 10
max_value = 100 # Upper bound in range.random()

input_length = sample_length + 1 # For special chars
vocab_size = max_value + 2 # +2 for SOS and EOS

def main() -> None:
    print('Generating Dataset')
    trainEncoderInput, trainDecoderInput, trainDecoderOutput = generateDataset(num_samples, sample_length, max_value, vocab_size)

    valEncoderInput, valDecoderInput, valDecoderOutput = generateDataset(num_samples, sample_length, max_value, vocab_size)
    print(trainEncoderInput)
    print(trainDecoderInput)
    print(trainDecoderOutput)
    print('Dataset Generated!')

    modelName = "pointer"

    model = model_factory(modelName, vocab_size, input_length, embedding_dims, lstm_units)
    model.summary()
    # dataset = ArtificialDataset(10)
    
    # for v in dataset._generator(2, 10):
    #    print(v)

    model.fit(
        x=[trainEncoderInput, trainDecoderInput],
        y=trainDecoderOutput,
        epochs=20,
        batch_size=128,
        shuffle=True,
        validation_data=([valEncoderInput, valDecoderInput], valDecoderOutput),
        # callbacks = [tensorboard_callback]
    )

    return 1

if __name__ == "__main__":
    main()
