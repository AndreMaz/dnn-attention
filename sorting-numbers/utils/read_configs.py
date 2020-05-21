import json


def get_configs(argv):    
    # Load the JSON with the configs for the selected environment
    with open(f"./sorting-numbers/config.json") as json_file:
        configs = json.load(json_file)

    # +3 for MASK, SOS and EOS
    configs['vocab_size'] = configs['max_value'] + 3
    # max_value is not included in range(max_value)
    configs['EOS_CODE'] = configs['max_value'] + 1
    # next element
    configs['SOS_CODE'] = configs['max_value'] + 2
    # +1 for the SOS/EOS symbol at the begginning
    configs['input_length'] = configs['sample_length'] + 1


    # Get model name from args
    try:
        modelName = sys.argv[1]
    except:
        # Use pointer by default
        modelName = 'pointer-masking'

    # Store the model name
    configs['model_name'] = modelName


    print('Configs:')
    for key in configs.keys(): print(f"{key}: {configs[key]}")

    return configs