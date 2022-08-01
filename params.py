import csv

class Params:
    '''
    A place to store all the parameters for the model
    '''
    BATCH_SIZE = 128
    NUM_CLASSES = 10
    DEVICE = 'cuda'
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    EPOCHS = 164

    @classmethod
    def save_params_to_csv(cls, filename: str, **kwargs):
        # get the class variables
        class_vars = vars(cls)
        class_var_dict = {k.lower(): v for k, v in class_vars.items() if not k.startswith('__') and k.isupper()}
        # combine with kwargs
        class_var_dict.update(kwargs)
        # write to csv
        with open(filename, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=class_var_dict.keys())
            writer.writeheader()
            writer.writerow(class_var_dict)
