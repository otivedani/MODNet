import numpy as np
import numpy.typing as npt
from typing import Union, Callable
from scipy.interpolate import interp1d


def adjust_curves(image: np.ndarray,
                  points: npt.ArrayLike,
                  mode: str = 'precise',
                  copy: bool = True,
                  ) -> np.ndarray:
    """Apply curves based on points to vectors

    Parameters
    ----------
    image : ndarray
        image to be applied with curves function
        can be used for other than image
        array shape is (any, channel)
    points : array_like
        1 or 3 or 4 list of curve points
            1 - equal adjust to all channel
            3 - respective adjust to each channel
            4 - respective adjust each and all channel
        points shape is ({1,3,4}, any, 2)
    mode : {'precise', 'closed', 'extrapolate'}, optional
        interpolation method for fitting curves to points
            'precise' - unclosed, out of x will be filled with first-last
            'closed' - closed, extrapolate to 0,0 and 255,255
            'extrapolate' - unclosed, extrapolate to outer range (then clipped)

    Returns
    -------
        copy of input ndarray, each channel applied with each curves
    """
    _CURVES_STRATS_ = {1, 3, 4}
    _CURVES_MODES_ = {
        'precise': _fit_precise,
        'closed': _fit_closed,
        'extrapolate': _fit_extrapolate,
    }

    strat = len(points)
    if strat not in _CURVES_STRATS_:
        raise ValueError('points length must either in {}'.format(_CURVES_STRATS_))
    if mode not in _CURVES_MODES_.keys():
        raise ValueError('curves mode must either in {}'.format(_CURVES_MODES_.keys()))

    # points to interpolate
    curves_f = [_CURVES_MODES_[mode](p) for p in points]

    # prepare output
    out_image = image.copy() if copy else image

    # case 3 : apply to each channel separately (c0, c1, c2)->(R, G, B)
    # case 1 : apply to all channel equally (c0)->(RGB)
    # case 4 : apply 3 + 1 (c0, c1, c2)->(R, G, B) ; (c3)->(RGB)
    _apply_each = strat in {3, 4}
    _apply_all = strat in {1, 4}

    if (_apply_each):
        for i, curve_f in enumerate(curves_f[:3]):
            out_image[..., i] = np.clip(curve_f(out_image[..., i]), 0, 255)

    if (_apply_all):
        out_image = np.clip(curves_f[-1](out_image), 0, 255)

    # applied curves
    return out_image


def _prep_pairs(p: npt.ArrayLike, close_range: Union[tuple, None] = None) -> np.ndarray:
    """Prepare pair of points"""
    # create and check
    pt = np.asarray(p, dtype=np.uint8)
    if pt.shape[-1] != 2:
        raise ValueError('list not containing pairs of points')
    # sort and remove duplicates in x
    _pt, i = np.unique(pt[:, 0], return_index=True)
    pt = pt[i]

    if close_range is not None:  # begin with 0, end with 255
        if _pt[0] != 0:
            pt = np.vstack(([close_range[0], close_range[0]], pt))
        if _pt[-1] != 255:
            pt = np.vstack((pt, [close_range[1], close_range[1]]))

    return pt


def _fit_precise(points: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Create functions from interpolating points, unclosed, fill with y_first, y_last"""
    pt_v = _prep_pairs(points)
    return interp1d(pt_v[:, 0], pt_v[:, 1],
                    kind=('quadratic' if pt_v.shape[0] > 2 else 'linear'),
                    bounds_error=False, fill_value=(pt_v[0, 1], pt_v[-1, 1]))


def _fit_closed(points: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Create functions from interpolating points, closed bounds, extrapolate to bounds"""
    pt_v = _prep_pairs(points, close_range=(0, 255))
    return interp1d(pt_v[:, 0], pt_v[:, 1],
                    kind=('quadratic' if pt_v.shape[0] > 2 else 'linear'),
                    bounds_error=False, fill_value=(pt_v[0, 1], pt_v[-1, 1]))


def _fit_extrapolate(points: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Create functions from interpolating points, unclosed, extrapolate to outer bounds"""
    pt_v = _prep_pairs(points)
    return interp1d(pt_v[:, 0], pt_v[:, 1],
                    kind=('quadratic' if pt_v.shape[0] > 2 else 'linear'),
                    fill_value='extrapolate')
