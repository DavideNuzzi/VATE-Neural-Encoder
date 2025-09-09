import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go
from matplotlib.gridspec import GridSpec

def plot_neural_data(data, fig=None):

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    T, C = data.shape

    if fig is None:
        fig_height = min(25, C * 1.5)
        fig = plt.figure(figsize=(10, fig_height))

    channel_separation = np.mean(np.percentile(data, 90, axis=0) - np.percentile(data, 10, axis=0))

    for i in range(C):
        plt.plot(data[:,i] + 5 * channel_separation * i, color='k', linewidth=1)

    plt.grid(True, linestyle=':', alpha=0.5)
    plt.xlabel('Time step')
    plt.yticks([5 * channel_separation * i for i in range(C)], [f'Channel {i}' for i in range(C)])



def plot_latent_trajectory(z_latent, label=None, ax=None, alpha=0.5, markersize=0):

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(projection='3d', azim=120)    

    # Make sure that the data is on the cpu and has the right shape
    if isinstance(z_latent, torch.Tensor):
        z_latent = z_latent.detach().cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()
    
    if z_latent.ndim > 2:
        z_latent = z_latent.squeeze()
    
    # If the labels are provided, handle them (for now only DISCRETE)
    if label is not None:
        if label.ndim > 1:
            label = label.squeeze()

        # Split the trajectory into segments based on labels
        changed_label_inds = list(np.where(label[1:] != label[0:-1])[0])

        # Also add an ind corresponding to the last point to make sure the last 
        # segment is plotted
        changed_label_inds.append(z_latent.shape[0]-1)

    else:
        # If there are no labels, fake it
        changed_label_inds = [z_latent.shape[0]-1]
        label = np.zeros(z_latent.shape[0], dtype=np.int32)

    prev_ind = 0

    for ind in changed_label_inds:

        l = label[ind]
        x = z_latent[prev_ind:ind+1, :]
        col = colors[l]

        ax.plot(*x.T, lw=0.5, color=col, marker='.', markersize=markersize, alpha=alpha)

        if ind < z_latent.shape[0] - 2:
            ax.plot([z_latent[ind+1, 0], z_latent[ind+2, 0]],
                    [z_latent[ind+1, 1], z_latent[ind+2, 1]],
                    [z_latent[ind+1, 2], z_latent[ind+2, 2]],
                    lw=0.5, color=col, alpha=alpha)

        prev_ind = ind
        


def plot_latent_trajectory_multiview(z_latent, label=None, views=[120]):

    cols = min(len(views), 2)
    rows = len(views) // cols + 1

    fig = plt.figure(figsize=(cols*5, rows*5))
    
    for i in range(len(views)):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d', azim=views[i])
        plot_latent_trajectory(z_latent, label, ax)


def animate_latent_trajectory(z_latent, label, savepath,
                              show_trajectory_point=False, point_speed_fps=30, point_size=5,
                                seconds=4, camera_rotations_per_second=1):

    fps = 30
    frames = seconds * fps

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection='3d', azim=120)

    # Make sure that the data is on the cpu and has the right shape
    if isinstance(z_latent, torch.Tensor):
        z_latent = z_latent.detach().cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()
        
    if z_latent.ndim > 2:
        z_latent = z_latent.squeeze()

    # Plot the trajectory
    plot_latent_trajectory(z_latent, label, ax)

    # If needed, plot a point on it
    if show_trajectory_point:
        point, = ax.plot(z_latent[0,0], z_latent[0,1], z_latent[0,2], '.', markersize=point_size, color='r')
        
    def animate(i):
        s = i / frames * seconds
        ax.view_init(30, s * camera_rotations_per_second * 360, 0)

        if show_trajectory_point:

            k = int((s * point_speed_fps) % z_latent.shape[0])
        
            point.set_data([z_latent[k,0]], [z_latent[k,1]])
            point.set_3d_properties([z_latent[k,2]])


        return (ax, point, )

    ani = animation.FuncAnimation(fig, animate, repeat=True, frames=fps*seconds, interval=1.0/fps)
    writer = animation.PillowWriter(fps=fps, metadata=dict(artist='Me'), bitrate=500)
    ani.save(savepath, writer=writer)


def plot_latent_trajectory_interactive(z_latent, label=None, alpha=0.5, markersize=2):
    if isinstance(z_latent, torch.Tensor):
        z_latent = z_latent.detach().cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()

    if z_latent.ndim > 2:
        z_latent = z_latent.squeeze()
    if label is not None and label.ndim > 1:
        label = label.squeeze()

    if label is None:
        label = np.zeros(z_latent.shape[0], dtype=np.int32)

    fig = go.Figure()

    # Assign a consistent color per label using matplotlib's color cycle
    unique_labels = np.unique(label)
    color_map = {l: plt.cm.tab10(i % 10) for i, l in enumerate(unique_labels)}  # RGB in [0,1]
    color_map = {l: f'rgb({int(255*c[0])},{int(255*c[1])},{int(255*c[2])})' for l, c in color_map.items()}

    # Track which labels already had a legend entry
    legend_shown = set()

    # Identify segment boundaries
    changed_label_inds = list(np.where(label[1:] != label[:-1])[0])
    changed_label_inds.append(len(label) - 1)

    prev_ind = 0
    for ind in changed_label_inds:
        l = label[ind]
        x = z_latent[prev_ind:ind + 1, :]

        fig.add_trace(go.Scatter3d(
            x=x[:, 0], y=x[:, 1], z=x[:, 2],
            mode='lines+markers' if markersize > 0 else 'lines',
            marker=dict(size=markersize, opacity=alpha, color=color_map[l]),
            line=dict(width=1, color=color_map[l]),
            name=f'label {l}',
            showlegend=(l not in legend_shown)
        ))

        legend_shown.add(l)
        prev_ind = ind

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title='Latent Trajectory (Interactive)',
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()



def plot_transition_model(model, x_vec, y_vec, z_vec):

    # Get the vector field associated with the transition model
    device = next(model.parameters()).device

    # Creo la griglia di punti
    xv, yv, zv = np.meshgrid(x_vec, y_vec, z_vec, indexing='ij')

    xv = xv.reshape(-1, 1)
    yv = yv.reshape(-1, 1)
    zv = zv.reshape(-1, 1)

    points = torch.tensor(np.concatenate((xv, yv, zv), axis=1), dtype=torch.float32, device=device)

    model.eval()
    dz = model.transition(points).detach().cpu().numpy()

    # Plot it
    plot_vector_field_interactive(xv, yv, zv, dz)



def plot_vector_field_interactive(x, y, z, vec_field, arrow_size=1, arrow_color='black'):
    """
    Plot a 3D interactive vector field using Plotly.
    """

    fig = go.Figure(data=go.Cone(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        u=vec_field[:,0], v=vec_field[:,1], w=vec_field[:,2],
        sizemode="scaled",
        sizeref=arrow_size,
        anchor="tail",
        showscale=False,
        colorscale=[[0, arrow_color], [1, arrow_color]]
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='cube'
        ),
        title="Interactive 3D Vector Field",
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()


   

def training_summary_plot(model, dataset, losses, label_to_show=None, title_args=dict()):

    # Training loss, top left
    fig = plt.figure(figsize=(8.27, 11.69), constrained_layout=True)


    gs = GridSpec(6, 6, figure=fig)

    # Training losses
    ax_losses = fig.add_subplot(gs[0,0:3])

    for k in losses:
        ax_losses.plot(losses[k], label=k)

    ylim = np.array(ax_losses.get_ylim())
    if ylim[0] < -100: ylim[0] = -100
    if ylim[1] > 100: ylim[1] = 100

    ax_losses.set_ylim(ylim)

    ax_losses.grid(True, linestyle=':', alpha=0.5)
    ax_losses.legend(fontsize=9)
    ax_losses.tick_params(labelsize=8)
    ax_losses.set_xlabel('Iteration', fontsize=9)
    ax_losses.set_ylabel('Loss', fontsize=9)
    ax_losses.set_title('Training losses', fontsize=10)

    # Training encoding
    x = dataset.neural.data.unsqueeze(0).unsqueeze(0)

    z_encoded_train, _, _ = model.encoder(x)
    z_encoded_train = z_encoded_train.squeeze()

    if label_to_show is not None:
        label = dataset.target_labels[label_to_show].data
    else:
        label = None

    ax_training_enc_0 = fig.add_subplot(gs[2:4,0:3], projection='3d', azim=0)
    plot_latent_trajectory(z_encoded_train, label=label, ax=ax_training_enc_0, markersize=1)
    ax_training_enc_0.set_title('Training set encoding (view angle = 0°)', fontsize=10)

    ax_training_enc_1 = fig.add_subplot(gs[2:4,3:], projection='3d', azim=90)
    plot_latent_trajectory(z_encoded_train, label=label, ax=ax_training_enc_1, markersize=1)
    ax_training_enc_1.set_title('Training set encoding (view angle = 90°)', fontsize=10)

    ax_training_enc_2 = fig.add_subplot(gs[4:,0:3], projection='3d', azim=180)
    plot_latent_trajectory(z_encoded_train, label=label, ax=ax_training_enc_2, markersize=1)
    ax_training_enc_2.set_title('Training set encoding (view angle = 180°)', fontsize=10)

    ax_training_enc_3 = fig.add_subplot(gs[4:,3:], projection='3d', azim=270)
    plot_latent_trajectory(z_encoded_train, label=label, ax=ax_training_enc_3, markersize=1)
    ax_training_enc_3.set_title('Training set encoding (view angle = 270°)', fontsize=10)

    # Generative model
    generation_result = model.generate(x0=dataset.neural.data[0,:].unsqueeze(0).unsqueeze(0), num_steps=5000)
    z_rollout, target_labels_rollout = generation_result['z_sequence'], generation_result['target_preds']
    target_label_rollout_class = None
    if label_to_show is not None:
        target_label_rollout_class = torch.argmax(target_labels_rollout[label_to_show], axis=-1)

    ax_gen = fig.add_subplot(gs[0:2,3:], projection='3d', azim=0)
    plot_latent_trajectory(z_rollout, label=target_label_rollout_class, ax=ax_gen, markersize=1, alpha=1)

    if z_rollout.abs().max() > 100 or torch.isnan(z_rollout.abs().max()) or torch.isinf(z_rollout.abs().max()):
        ax_gen.set_title('Generative rollout (divergence detected)', fontsize=10)
        
        limit_ind = torch.where(z_rollout.abs() > 100)[0][0]
        z_limits_min = z_rollout[0:limit_ind,:].min(dim=0).values.detach().cpu().numpy()
        z_limits_max = z_rollout[0:limit_ind,:].max(dim=0).values.detach().cpu().numpy()
        ax_gen.set_xlim([z_limits_min[0], z_limits_max[0]])
        ax_gen.set_ylim([z_limits_min[1], z_limits_max[1]])
        ax_gen.set_zlim([z_limits_min[2], z_limits_max[2]])
        divergence_detected = True
    else:
        divergence_detected = False
        ax_gen.set_title('Generative rollout', fontsize=10)
 
    # Generative model plain
    ax_gen_lines = fig.add_subplot(gs[1,0:3])
    ax_gen_lines.plot(z_rollout.detach().cpu().numpy())
    ax_gen_lines.grid(True, linestyle=':', alpha=0.5)
    ax_gen_lines.tick_params(labelsize=8)
    ax_gen_lines.set_xlabel('Time', fontsize=9)
    ax_gen_lines.set_ylabel('z component', fontsize=9)
    ax_gen_lines.set_title('Generative rollout trajectories', fontsize=10)

    # Title and stats
    for k in losses:
        title_args[f'{k}_last'] = np.mean(losses[k][-100:])

    title_str = ''
    n = 0
    for k in title_args:
        v = title_args[k]
        param_str = f'{k}: '
        if isinstance(v, float):
            param_str += f'{v:.4f}'
        else:
            param_str += f'{v}'
        n += 1

        title_str += param_str
        if n % 3 == 0:
            title_str += '\n'
        else:
            title_str += '   |   '

    fig.suptitle(title_str, fontsize=10)