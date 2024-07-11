from deepfinder.models.res_unet import my_res_unet_model
from deepfinder.models.unet import my_unet_model 

def load_model(dim_in, Ncl, model_name, trained_weights_path):

    # Play with Model to Use
    assert model_name in ['unet', 'res_unet'], "Invalid model name specified. Use 'unet' or 'res_unet'."

    if model_name == 'unet':
        net = my_unet_model(dim_in, Ncl)
    elif model_name == 'res_unet':
        net = my_res_unet_model(dim_in, Ncl)
    else:
        raise ValueError("Invalid model name specified. Valid options {unet, or res_unet}")

    if trained_weights_path is not None:
        net.load_weights(trained_weights_path)
        print(f'Loading {model_name} with Pre-Train Weights')
    else:
        print(f'Loading {model_name} without Loading Pre-Train Weights')        

    return net