from dataclasses import dataclass


@dataclass
class Params:
    nx: int
    nz: int
    xlen: float
    zlen: float
    nvariables: int
    ntimesteps: int
    variable_index: int
    directory: str
    plot_nlevels: int
    plot_nticks: int
    plot_no_colorbar: bool
    filename: str


def make_dataclass_from_args(args):
    return Params(**vars(args))
