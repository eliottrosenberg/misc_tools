import scipy
import numpy as np


def _find_lower_index(index: int, indices_to_keep: np.ndarray) -> int:
  for j in reversed(range(index)):
    if indices_to_keep[j]:
      return j

def sparsify(x_data: np.ndarray, y_data: np.ndarray, y_tol: float) -> tuple[np.ndarray, np.ndarray]:
  """Sparsify the input data so that it is well approximated by a linear interpolation of the output data

  Args:
    x_data: The dense input x-coordinates
    y_data: The dense input y-coordinates
    y_tol: The tolerance for errors in the y direction

  Returns:
    The sparsified x and y data.
  """
  indices_to_keep = np.ones(len(x_data), dtype=bool)
  assert len(x_data) == len(y_data)
  for index in range(1,len(x_data)-1):
    indices_to_keep[index] = False
    x_sparse = x_data[indices_to_keep]
    y_sparse = y_data[indices_to_keep]
    interp_function = scipy.interpolate.interp1d(x_sparse, y_sparse)

    ## condition to keep this sparsification:
    keep_sparsify = True
    lower_index = _find_lower_index(index, indices_to_keep)
    upper_index = index+1
    for j in range(lower_index+1, upper_index):
      x = x_data[j]
      y = y_data[j]
      if abs(interp_function(x) - y) > y_tol:
        keep_sparsify = False
        break
    if not keep_sparsify:
      indices_to_keep[index] = True

  return x_data[indices_to_keep], y_data[indices_to_keep], indices_to_keep
