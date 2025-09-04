import pickle as pkl
from pathlib import Path
from utils.plotting import plot_latent_trajectory
from matplotlib import pyplot as plt


def plot_latent_space_callback(epoch, model, x_neural, x_labels, savepath, epochs_skip=100):

    if not savepath.exists():
        Path.mkdir(savepath, parents=True, exist_ok=True)

    if (epoch + 1) % epochs_skip == 0:
        savefile = savepath / f'epoch_{epoch + 1}.png'

        z_latent = model.encoder(x_neural).squeeze(0).detach().cpu()

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(projection='3d')
        plot_latent_trajectory(z_latent, x_labels, ax)
        plt.savefig(savefile)
        plt.close()



# def log_epoch_callback(epoch, losses, savepath, epochs_skip=100):

#     # if savepath is None:
#     #     savepath = Path.cwd() / 'logs'
#     #     experiment_num = len(list(savepath.glob('*')))
#     #     savepath = savepath / f'experiment_{experiment_num}'

#     if not savepath.exists():
#         Path.mkdir(savepath, parents=True, exist_ok=True)

#     if (epoch + 1) % epochs_skip == 0:
#         savefile = savepath / f'epoch_{epoch + 1}.pkl'

#         with open(savefile, 'wb') as f:
#             pkl.dump(losses, f)
    