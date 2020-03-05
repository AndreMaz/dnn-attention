# from dataset.dataset_generator import ArtificialDataset
from dataset.generator import generateEncoderInput
from models.model_factory import model_factory

embeddingDims = 64
lstmUnits = 64

num_samples = 1
sample_length = 10
max_value = 10
vocab_size = max_value + 1 # +1 For start of Sequence

def main() -> None:
    print('Generating Dataset')
    trainEncoderInput, trainDecoderInput, trainDecoderOutput = generateEncoderInput(num_samples, sample_length, max_value, vocab_size)
    print(trainEncoderInput)
    print(trainDecoderInput)
    print(trainDecoderOutput)
    print('Dataset Generated!')

    modelName = "pointer"

    model = model_factory(modelName, vocab_size, sample_length, embeddingDims, lstmUnits)
    model.summary()
    # dataset = ArtificialDataset(10)
    
    # for v in dataset._generator(2, 10):
    #    print(v)

    model.fit(
        x=[trainEncoderInput, trainDecoderInput],
        y=trainDecoderOutput,
        epochs=3,
        batch_size=128,
        shuffle=True,
        # validation_data=([valEncoderInput, valDecoderInput], valDecoderOutput),
        # callbacks = [tensorboard_callback]
    )

    return 1

if __name__ == "__main__":
    main()
