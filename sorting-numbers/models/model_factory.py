from models.pointer_network.model import createModel as pointerModel
from models.pointer_masking.model import createModel as pointerMasking

models = {
    'pointer': pointerModel,
    'pointer-masking': pointerMasking
}

def model_factory(name, vocab_size, input_length, embedding_dims, lstm_units):
    print(name)
    try:
        createModel = models[name]
        print(f'Using "{name.upper()}" model')
        return createModel(vocab_size, input_length, embedding_dims, lstm_units)
    except:
        raise NameError('Unknown Model Name! Select one of: "pointer"')


if __name__ == "__main__":
    pass