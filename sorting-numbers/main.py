# from dataset.dataset_generator import ArtificialDataset
from dataset.generator import generateDataset
from models.model_factory import model_factory
from models.inference import runSeq2SeqInference
import tensorflow as tf
import numpy as np

embedding_dims = 64
lstm_units = 64

num_samples_training = 150_000
num_sample_validation = 50_000

sample_length = 10
max_value = 100 # Upper bound in range.random()

vocab_size = max_value + 2 # +2 for SOS and EOS
input_length = sample_length + 1 # For special chars at the beggining of input

def main() -> None:
    print('Generating Dataset')
    trainEncoderInput, trainDecoderInput, trainDecoderOutput = generateDataset(num_samples_training, sample_length, max_value, vocab_size)

    valEncoderInput, valDecoderInput, valDecoderOutput = generateDataset(num_sample_validation, sample_length, max_value, vocab_size)
    print(trainEncoderInput)
    print(trainDecoderInput)
    print(trainDecoderOutput)
    print('Dataset Generated!')

    modelName = "pointer"

    model = model_factory(modelName, vocab_size, input_length, embedding_dims, lstm_units)
    model.summary()
    
    model.fit(
        x=[trainEncoderInput, trainDecoderInput],
        y=trainDecoderOutput,
        epochs=15,
        batch_size=128,
        shuffle=True,
        validation_data=([valEncoderInput, valDecoderInput], valDecoderOutput),
        # callbacks = [tensorboard_callback]
    )

    # # Test the model
    num_samples_tests = 200
    correctPredictions = 0
    wrongPredictions = 0
    trainEncoderInput, _, _ = generateDataset(num_samples_tests, sample_length, max_value, vocab_size)
    for _, inputEntry in enumerate(trainEncoderInput):
        print('__________________________________________________')

        print(list(inputEntry.numpy().astype("int16")[1:]))

        # Generate correct answer
        correctAnswer = list(inputEntry.numpy().astype("int16"))
        # Remove EOS
        correctAnswer = correctAnswer[1:]
        correctAnswer.sort()

        print(correctAnswer)

        # Add the batch dimension [batch=1, features]
        inputEntry = tf.expand_dims(inputEntry, 0)

        predictedAnswer = runSeq2SeqInference(model, inputEntry, vocab_size, input_length, max_value)
        print(predictedAnswer)

        diff = []
        for index, value in enumerate(correctAnswer):
            diff.append(correctAnswer[index] - predictedAnswer[index])
        
    
        if (all(result== 0 for (result) in diff)):
            correctPredictions += 1
            print('______________________OK__________________________')
        else:
            wrongPredictions += 1
            print('_____________________WRONG!_______________________')
    
    print(f"Correct Predictions: {correctPredictions/num_samples_tests} || Wrong Predictions: {wrongPredictions/num_samples_tests}")

if __name__ == "__main__":
    main()
