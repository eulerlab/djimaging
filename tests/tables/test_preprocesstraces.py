import numpy as np

from djimaging.tables.core.preprocesstraces import drop_left_and_right


def _gen_outlier_trace(nleft, nright, n=100, cval=-5., cval_right=None):
    trace = np.random.uniform(0, 1, n)

    if cval_right is None:
        cval_right = cval

    if nleft > 0:
        trace[:nleft] = cval
    if nright > 0:
        trace[-nright:] = cval_right
    return trace


def test_drop_left_and_right_remove_single_outliers():
    trace = _gen_outlier_trace(nleft=1, nright=1, cval=-5., n=20)
    trace = drop_left_and_right(trace, drop_nmin_lr=(0, 0), drop_nmax_lr=(3, 3), inplace=False)
    assert np.all(trace >= 0)


def test_drop_left_and_right_remove_multiple_outliers():
    trace = _gen_outlier_trace(nleft=3, nright=3, cval=-5., n=20)
    trace = drop_left_and_right(trace, drop_nmin_lr=(0, 0), drop_nmax_lr=(3, 3), inplace=False)
    assert np.all(trace >= 0)


def test_drop_left_and_right_remove_only_left_outliers():
    trace = _gen_outlier_trace(nleft=3, nright=3, cval=-5., cval_right=-1., n=20)
    trace = drop_left_and_right(trace, drop_nmin_lr=(0, 0), drop_nmax_lr=(3, 0), inplace=False)
    assert trace[3] >= 0
    assert np.all(trace[:3] == trace[3])
    assert np.all(trace[-3:] == -1.)
