import os

if "LEGATE_MAX_DIM" in os.environ and "LEGATE_MAX_FIELDS" in os.environ:
    from legate.timing import time
else:
    import time as t

    def time():
        return t.time_ns() / 1e3


class TimedCodeBlock(object):
    """Use this as a context manager to time a block
    of code. For example:

    with TimedCodeBlock(label="Elapsed time for initialization"):
        z = x + y

    will print the label and return the elapsed time in s:
    Elapsed time for initialization: x s

    """

    def __init__(self, label: str = "Elapsed time"):
        self.label = label

    def __enter__(self):
        self.elapsed_time = time()
        return self.elapsed_time

    def __exit__(self, type, value, traceback):
        self.elapsed_time = (time() - self.elapsed_time) / 1e6
        print(f"{self.label}: {self.elapsed_time} s", flush=True)
