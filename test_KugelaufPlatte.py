from __future__ import annotations

import pytest
import scipy
import matplotlib.pylab as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from pylab.core import timeseries
import numpy
import numpy as np
from numpy import trapz
from scipy import integrate
from pylab.core import loader
from pylab.core import testing
from pylab.simulink import simulink

from pylab.simulink import _engine


_engine.import_matlab_engine("R2021b")


def test_experiment():
    info = loader.load_test("test.yml")
    details = simulink.load_details("matlab_detail.yml")
    # details.devices[0].data["params"]["Kr"] = 0.6
    # details.devices[0].data["params"]["D"][1] = 0.2
    # print(details.devices[0].data["params"])
    experiment = simulink.create(info, details)
    report = experiment.execute()
    if report.failed:
        raise AssertionError(report.what)
    result = report.results["PT1Regler.Alpha"]
    s = result.pretty_string()
    a = result.values
    asign = np.sign(a)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    A = signchange
    G = np.where(A == 1)
    print(G)
    T = result.time
    print(T[G[0]])
    finterp = InterpolatedUnivariateSpline(T, a, k=1)
    qq = [finterp.integral(0, t) for t in T
    print(qq)
    # print(s)
    # A = result.values
    # print(A)
    # zero_crossings = numpy.where(numpy.diff(numpy.signbit(A)))[0]
    # print(zero_crossings)
    # for i in A:
    #     if abs(A[i])>1e-8 and abs(A[i+1])>1e-8:
    #         continue
    #         if A[i]*A[i+1]<0:
    #            print(A[i])

    # ueberschwingweite = max(result.values)
    # Schwingweite = max(result.values)+abs(min(result.values))
    # print("Schwingweite:",float(Schwingweite))
    # print("Überschwingweite:",floa(uerberschwingweite))
    # def zero_crossings(ts: TimeSeries, epsilon: float = 1e-8) -> int:
    #     """Return the number of zero-crossings of a timeseries.
    #
    #     Args:
    #         ts: The time series to check
    #         epsilon: The absolute tolerance
    #     """
    #     # TODO Assert that the time series is scalar-valued.
    # count = 0
    # previous_sign = None
    # for v in result.values:
    #         # If the value is too close to zero, just ignore it.
    #     if abs(v) < 1e-8:
    #         continue
    #         current_sign = (v > 0)
    #         # Note that this branch never happens when the sign is still
    #         # indeterminate
    #         if current_sign != previous_sign:
    #             count += 1
    #             previous_sign = current_sign
    # return count
    # print(count)
    # z = zero_crossings(result, 1e-8)
    # print(z)
    # T = result.time
    # Z = []
    # for i in result.values:
    #     if i > 1e-8:
    #         continue
    #         print(i)
    #     #     print(i)
    #     # if abs(i+1) > 1e-8:
    #     #     continue
    #     #     print(i+1)
    #     #     if i*i+1 < 0:
    #     #        print(i)
    # def zero_(x, epsilon):
    #     z = []
    #     for v in x:
    #         if abs(v) > epsilon:
    #             continue
    #             if abs(v + 1) > epsilon:
    #                 continue
    #                 if v * (v + 1) < 0:
    #                     z.append(v)
    #     return z
    # Z = zero_(result.values, 1e-8)
    # print(Z)
    # f = result.interpolate(kind="linear")
    # plt.plot(result.time,result.values)
    #
    # x = result.time
    # y = result.values
    # g = UnivariateSpline(x, y)

    # h = lambda x: abs(f(x))
    # energie = scipy.integrate.quad(h, 0, 10)
    # print(energie)
    #
    #
    # f = result.interpolate(kind="linear")
    # assert abs(f(3)) < 0.01
    # expected = timeseries.TimeSeries(
    #     [3, 4, 5, 6, 7, 8, 9, 10],
    #     [[0], [0], [0], [0], [0], [0], [0], [0]],
    # )
    # g = expected.interpolate(kind="linear")
    # timeseries.assert_almost_everywhere_close(
    #     f,
    #     g,
    #     3,
    #     10,
    #     atol = 0.01
    # )
