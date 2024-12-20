import pathlib
import h5py
import numpy as np

def return_data(fileName: pathlib.Path, data: str) -> np.ndarray:
    with h5py.File(fileName, "r") as logfile:
        data = logfile[data][:]
    return data
