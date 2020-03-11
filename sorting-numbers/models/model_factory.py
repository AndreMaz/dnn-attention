from models.pointer_network.model import createModel as pointerModel
from models.pointer_masking.model import createModel as pointerMasking

models = {
    'pointer': pointerModel,
    'pointer-masking': pointerMasking
}

def model_factory(name, vocab_size, input_length, embedding_dims, lstm_units):
    try:
        createModel = models[name]
        print(f'Using "{name.upper()}" model')
    except:
        raise NameError('Unknown Model Name! Select one of: "pointer"')
    
    return createModel(vocab_size, input_length, embedding_dims, lstm_units)


if __name__ == "__main__":
    pass