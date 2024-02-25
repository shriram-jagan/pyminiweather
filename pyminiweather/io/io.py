import pathlib

from pyminiweather import numpy as np
from pyminiweather.data import Fields


class Writer:
    def warn_if_file_exists(file: pathlib.Path) -> bool:
        return True if file.exists() else False

    def open_file(filename: str, permissions: str):
        path = pathlib.Path(filename)
        Writer.warn_if_file_exists(path)
        return open(path, permissions)

    @staticmethod
    def write_array(
        filename: str,
        array: np.ndarray,
        delimiter: str = ",",
        permissions: str = "ab",
    ):
        """
        The arrays can have dimensions greater than 2.
        """

        def _write_4d_array(out_file, array, delimiter):
            dim = array.shape
            for dim_0 in range(dim[0]):
                for dim_1 in range(dim[1]):
                    np.savetxt(
                        out_file,
                        array[dim_0][dim_1],
                        delimiter=delimiter,
                    )

        def _write_3d_array(out_file, array, delimiter):
            dim = array.shape
            for dim_0 in range(dim[0]):
                np.savetxt(out_file, array[dim_0], delimiter=delimiter)

        assert array.ndim <= 4

        out_file = Writer.open_file(filename, permissions)

        if array.ndim == 3:
            _write_3d_array(out_file, array, delimiter=delimiter)
        elif array.ndim == 4:
            _write_4d_array(out_file, array, delimiter=delimiter)
        else:
            np.savetxt(out_file, array, delimiter=delimiter)

    @staticmethod
    def write_state(
        filename,
        fields: Fields,
        delimiter: str = ",",
        permissions: str = "ab",
    ):
        """Write out the three-dimensional array state"""
        out_file = Writer.open_file(filename, permissions)
        for variable in range(fields.nvariables):
            np.savetxt(out_file, fields.state[variable], delimiter=delimiter)
