import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import Params, make_dataclass_from_args

from pyminiweather.mesh import MeshData

#plt.set_cmap("Spectral")


def get_parser():
    parser = argparse.ArgumentParser(
        description="This is a post-processing tool that makes images "
        "from the file written by PyMiniWeather. The images "
        "can then be concatenated to make a gif using ImageMagick. "
        "Make sure to set output-freq > 0 in PyMiniWeather "
        "and that a sub-directory named images exists within "
        "the directory passed to the script."
    )

    parser.add_argument(
        "--nx",
        type=int,
        default=200,
        dest="nx",
        help="Number of points in x-direction (default: 200)",
    )
    parser.add_argument(
        "--nz",
        type=int,
        default=100,
        dest="nz",
        help="Number of points in z-direction (default: 100)",
    )
    parser.add_argument(
        "--xlen",
        type=float,
        default=2e4,
        dest="xlen",
        help="Length of domain in x-direction (default: 2e4)",
    )
    parser.add_argument(
        "--zlen",
        type=float,
        default=1e4,
        dest="zlen",
        help="Length of domain in Z-direction (default: 1e4)",
    )
    parser.add_argument(
        "--nvariables",
        type=int,
        default=4,
        dest="nvariables",
        help="Number of variables in the simulation" " (default: 4)",
    )
    parser.add_argument(
        "--ntimesteps",
        type=int,
        default=10,
        dest="ntimesteps",
        help="Number of timesteps in the simulation" " (default: 10)",
    )
    parser.add_argument(
        "--variable-index",
        type=int,
        default=3,
        dest="variable_index",
        help="Variable index must be less than number of variables" " (default: 3)",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="./",
        dest="directory",
        help="Directory where the input file is located (default: ./)",
    )
    parser.add_argument(
        "--plot-nlevels",
        type=str,
        default=128,
        dest="plot_nlevels",
        help="Number of levels to be used in the plotter" " (default: 128)",
    )
    parser.add_argument(
        "--plot-nticks",
        type=str,
        default=10,
        dest="plot_nticks",
        help="Number of ticks to be used in the plotter" " (default: 10)",
    )
    parser.add_argument(
        "--plot-no-colorbar",
        action="store_true",
        default=False,
        dest="plot_no_colorbar",
        help="Use this flag to disable plotting the colorbar in the "
        "contour plots (default: False)",
    )
    parser.add_argument(
        "--plot-vmin-vmax",
        type=float,
        nargs=2,
        default=None,
        dest="plot_vmin_vmax",
        help="Provide the vmin and vmax for the plot",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="PyMiniWeatherData.txt",
        dest="filename",
        help="Name of the output file the solution variables"
        " will be written to. (default: PyMiniWeatherData.txt)",
    )

    return parser


# Get a dataclass from Arguments
parser = get_parser()
args, _ = parser.parse_known_args()
params = make_dataclass_from_args(args)
for k, v in params.__dict__.items():
    print(f"{k:25s} {v}")


# Create a mesh
mesh_params = {
    "nx": params.nx,
    "nz": params.nz,
    "xlen": params.xlen,
    "zlen": params.zlen,
    "dx": params.xlen / params.nx,
    "dz": params.zlen / params.nz,
    "hs": 2,
}
mesh = MeshData(mesh_params)

# sanity checks
image_directory = Path(params.directory) / "images"
if not image_directory.exists():
    warnings.warn(
        f"Make sure the images sub-directory inside"
        f" {params.directory} exists AND writeable"
    )

    sys.exit()

assert params.nvariables == 4
assert params.variable_index < params.nvariables

# Read the data file
base_filename = params.filename.split(".")[0]
py = np.loadtxt(Path(params.directory) / f"{base_filename}_svars.txt", delimiter=",").reshape(
    params.ntimesteps, params.nvariables, params.nz, params.nx
)

# create mesh
mesh_x, mesh_y = mesh.get_mesh_cell_centers()

# setup plotters
padding = 0.10
round_to_decimals = 2

if args.plot_vmin_vmax is None:
    vmin = py[0 : params.ntimesteps, params.variable_index].min()
    vmax = py[0 : params.ntimesteps, params.variable_index].max()
else:
    vmin, vmax = args.plot_vmin_vmax

print(f"Contour vmin/vmax: {vmin}, {vmax}")

levels = np.linspace(vmin, vmax, params.plot_nlevels)
ticks = np.linspace(vmin, vmax, params.plot_nticks).round(round_to_decimals)

step_start = 0
step_end = params.ntimesteps
step_skip = 1

# Loop through the timesteps and plot; this part can be accelerated using
# multiprocessing
for timestep in range(step_start, step_end, step_skip):
    fig = plt.figure()
    ax = plt.gca()
    plt.set_cmap("jet")

    quantity = py[timestep, params.variable_index]
    h = plt.contourf(mesh_x, mesh_y, quantity, levels=levels)
    plt.axis("scaled")

    # Make adjustments to fit the colorbar
    if not params.plot_no_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = plt.colorbar(h, cax=cax)

        # Adjust colorbar such that the ticks and ticklabels are not displayed
        cbar.ax.tick_params(size=0)

    # No ticks or ticklabels
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])

    plt.tight_layout()

    fname = f"{image_directory}/{timestep:0>3}.png"
    plt.savefig(fname, dpi=600)
    plt.close(fig)

# Make animation if requested
# TODO
