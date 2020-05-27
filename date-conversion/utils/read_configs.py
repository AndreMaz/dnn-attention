import json


def get_configs(argv):    
    toPlotAttention = False
    
    # Load the JSON with the configs for the selected environment
    with open(f"./date-conversion/config.json") as json_file:
        configs = json.load(json_file)

    # Get model name from args
    try:
        modelName = argv[1]
        toPlotAttention = True if int(argv[2]) == 1 else False
    except:
        # Use Luong by default
        modelName = 'luong'

    # Store the model name
    configs['model_name'] = modelName
    configs['to_plot_attention'] = toPlotAttention

    print('Configs:')
    for key in configs.keys(): print(f"{key}: {configs[key]}")

    return configs