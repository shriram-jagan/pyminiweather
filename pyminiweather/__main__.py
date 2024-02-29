import argparse

from pyminiweather import numpy as np
from pyminiweather.data import Fields, initialize_fields
from pyminiweather.ics import init
from pyminiweather.io import Writer
from pyminiweather.mesh import MeshData
from pyminiweather.post import compute_solution_variables, compute_stats
from pyminiweather.solve import evolve
from pyminiweather.utils import TimedCodeBlock, time


def get_parser():
    parser = argparse.ArgumentParser()
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
        help="Length of domain in x-direction (default: 2e4)."
        "Drastically changing default values will require "
        "changes in initial conditions",
    )
    parser.add_argument(
        "--zlen",
        type=float,
        default=1e4,
        dest="zlen",
        help="Length of domain in Z-direction (default: 1e4)"
        "Drastically changing default values will require "
        "changes in initial conditions",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=10,
        dest="nsteps",
        help="Number of time steps (default: 10)",
    )
    parser.add_argument(
        "--nwarmups",
        type=int,
        default=0,
        dest="nwarmups",
        help="Number of wamup time steps (default: 0)",
    )
    parser.add_argument(
        "--ic-type",
        type=str,
        default="thermal",
        choices=[
            "thermal",
            "collision",
            "gravity",
            "density-current",
            "injection",
        ],
        help="Type of initial condition. The default value is thermal.",
        dest="ic_type",
    )
    parser.add_argument(
        "--hs",
        type=int,
        default=2,
        choices=[
            2,
        ],
        dest="hs",
        help="This gives you an illusion of choice. "
        "This relates to the width of the stencil in the "
        "discretization and cannot be changed.",
    )
    parser.add_argument(
        "--s",
        type=int,
        default=4,
        choices=[
            4,
        ],
        dest="s",
        help="This gives you an illusion of choice. "
        "This relates to the size of interpolating "
        "kernel and cannot be changed.",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=500.0,
        dest="max_speed",
        help="Assumed maximum speed that is used in the computation of time-step (default: 500)",
    )
    parser.add_argument(
        "--cfl",
        type=float,
        default=1.00,
        dest="cfl",
        help="CFL condition that is used while computing the time-step."
        " (default: 1.0)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        dest="verbose",
        help="Set this to True to print detailed information about"
        " the simulation setup",
    )
    parser.add_argument(
        "--output-freq",
        type=int,
        default=-1,
        dest="output_freq",
        help="The solution varibales will be written to a file every "
        "output-freq steps but disabled by default (default: -1)",
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


def main():
    parser = get_parser()
    args, unknown_args = parser.parse_known_args()
    params = vars(args)

    params["dx"] = params["xlen"] / params["nx"]
    params["dz"] = params["zlen"] / params["nz"]
    params["dt"] = (
        np.minimum(params["dx"], params["dz"]) * params["cfl"] / params["max_speed"]
    )
    if params["verbose"]:
        for k, v in params.items():
            print(f"{k:25s} {v}")

    with TimedCodeBlock(label="Elapsed time for initialization"):
        fields = initialize_fields(params)
        mesh = MeshData(params)
        init(fields, params, mesh)

    total_mass_start, total_energy_start = compute_stats(params, fields)
    print(
        f"Start: total_mass, total_energy: {total_mass_start}, "
        f"{total_energy_start}",
        flush=True,
    )

    if params["nwarmups"]:
        # No I/O during warmups
        with TimedCodeBlock(label="Elapsed time for warmups"):
            for _ in range(params["nwarmups"]):
                evolve(params, fields, mesh, dt=params["dt"])

    with TimedCodeBlock(label="Elapsed time for timestepping"):
        for istep in range(params["nsteps"]):
            # I/O and print stats
            if params["output_freq"] > 0 and (istep + 1) % params["output_freq"] == 0:
                print(
                    f"Step: {istep}, max(rho*t): {fields.state[3].max()}",
                    flush=True,
                )
                fname, append = params["filename"].split(".")
                Writer.write_state(params["filename"], fields)
                Writer.write_array(
                    fname + "_svars." + append,
                    compute_solution_variables(params, fields),
                )

            # step through in time
            evolve(params, fields, mesh, dt=params["dt"])

    total_mass_end, total_energy_end = compute_stats(params, fields)
    print(
        f"End: total_mass, total_energy: {total_mass_end}, {total_energy_end}",
        flush=True,
    )

    total_mass_change = (total_mass_end - total_mass_start) / total_mass_start
    total_energy_change = (total_energy_end - total_energy_start) / total_energy_start

    print(
        f"Relative change in total_mass, total_energy: {total_mass_change}, "
        f"{total_energy_change}",
        flush=True,
    )


if __name__ == "__main__":
    main()
