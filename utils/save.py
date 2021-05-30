
import torch
import os

# ================================================================== #
# =================== Save model =================================== #
def save_model(checkpoint_folder, model, name=''):

    # create `models` folder if does not exists
    folder = compose_out_folder(checkpoint_folder, ['models'])

    file_name = 'model{}'.format(name)

    # save last model
    model_path = '{}/{}.pth'.format(folder, file_name)
    torch.save(model.state_dict(), model_path)

# ================================================================== #
# =================== Save surface map ============================= #
def save_meta_sample(checkpoint_folder, data, model, prefix=''):
    folder       = compose_out_folder(checkpoint_folder, ['sample']) # create `sample` folder if does not exists
    model_params = dict(model.named_parameters())

    dump_torch_sample(folder, model_params, data)


# ================================================================== #
# =================== Write binary data ============================ #
def dump_torch_sample(folder, model_weights, data):

    filename = '{}/flat_model.pth'.format(folder)
    sample     = {  'weights'      : model_weights,
                    **data # add data from sample
                    }

    torch.save(sample, filename)

# ================================================================== #
# =================== Create folder ================================ #
def compose_out_folder(checkpoint_dir, sub_folders):
    out_folder = os.path.join(checkpoint_dir, *sub_folders)
    os.makedirs(out_folder, exist_ok=True)

    return out_folder
# ================================================================== #
