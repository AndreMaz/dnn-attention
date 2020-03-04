# from dataset.dataset_generator import ArtificialDataset
from dataset.generator import generateEncoderInput
from models.model_factory import model_factory

embeddingDims = 64
lstmUnits = 64

inputLength = 10
vocabularySize = 100
numSamples = 50

def main() -> None:
    print('Generating Dataset')
    trainEncoderInput, trainDecoderInput, trainDecoderOutput = generateEncoderInput(numSamples, inputLength, vocabularySize)
    print('Dataset Generated!')

    modelName = "pointer"

    model = model_factory(modelName, vocabularySize+1, vocabularySize+1, inputLength, inputLength, embeddingDims, lstmUnits)
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
