import random
import torch.nn.functional as F

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def new_fig():
    """Create a new matplotlib figure containing one axis"""
    fig = Figure()
    FigureCanvas(fig)
    axes = []

    axes.append(fig.add_subplot(131))
    axes.append(fig.add_subplot(132))
    axes.append(fig.add_subplot(133))

    return fig, axes


def make_plot_val(inputs, outputs, labels):
    fig, ax = new_fig()

    idx = random.randint(0, inputs.shape[0]-1)

    input_ = inputs[idx].cpu().numpy().transpose(1,2,0)
    input_ = (input_ - input_.min()) / (input_.max() - input_.min())

    ax[0].imshow(input_)
    ax[0].set_title("Input")

    ax[1].imshow(labels[idx].cpu().numpy().squeeze(), vmin=0, vmax=46)
    ax[1].set_title("Ground Truth")

    out = F.softmax(outputs[idx], dim=0)
    out = out.argmax(dim=0)

    ax[2].imshow(out.detach().cpu().numpy().squeeze(), vmin=0, vmax=46)
    ax[2].set_title("Prediction")

    return fig
