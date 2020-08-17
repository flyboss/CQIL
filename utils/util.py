import torch

PAD_ID, UNK_ID = [0, 1]


def save_model(model, save_dir, epoch):
    print(f"\nitr_global: {epoch} save model")
    torch.save(model.state_dict(), f'{save_dir}epo{epoch}.h5')


def load_model(model, model_filepath):
    print(f'load model from {model_filepath}')
    model.load_state_dict(torch.load(model_filepath))
