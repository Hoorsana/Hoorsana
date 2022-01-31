from __future__ import annotations

import pytest
import scipy
import matplotlib.pylab as plt
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.integrate
from pylab.core import timeseries
import numpy
import numpy as np
from numpy import trapz
from scipy import integrate
from pylab.core import loader
from pylab.core import testing
from pylab.simulink import simulink
from collections import Counter
from collections import defaultdict
from pylab.simulink import _engine


_engine.import_matlab_engine("R2021b")


def test_experiment():
    info = loader.load_test("test.yml")
    details = simulink.load_details("matlab_detail.yml")
    details.devices[0].data["params"]["D"][1] = 0.2
    Kr = details.devices[0].data["params"]["Kr"]
    I = [r * 0.01 for r in range(0, 50, 5)]
    Kr_List = []
    Ueberschwing_Alpha = []
    Ueberschwing_X = []
    Schwing_Alpha = []
    Schwing_X = []
    E_Alpha = []
    E_X = []
    Z1 = []
    Z2 = []
    for i in I:
       details.devices[0].data["params"]["Kr"] = 0.6+i
       experiment = simulink.create(info, details)
       report = experiment.execute()
       if report.failed:
           raise AssertionError(report.what)
       print("Kr:", float(0.45+i))
       Kr_List.append(float(0.45+i))
                                       #TODO Alpha
       result = report.results["PT1Regler.Alpha"]
       alpha = result.pretty_string()
       alpha_V = result.values
       alpha_T = result.time
                                       #TODO Schwingweite Alpha
       Ueberschwing1 = max(alpha_V)
       Schwingweite1 = max(result.values)+abs(min(result.values))
       Ueberschwing_Alpha.append(Ueberschwing1)
       Schwing_Alpha.append(Schwingweite1)
                                       #TODO Energie von Alpha
       f_Alpha = InterpolatedUnivariateSpline(alpha_T, alpha_V, k=1)
       Energie_Alpha = scipy.integrate.quad_vec(lambda t: np.absolute(f_Alpha(t)), 0, alpha_T[-1])
       E_Alpha.append(Energie_Alpha)
                                      #TODO Zero Crossing

       z1 = timeseries.zero_crossings(result, 1e-9)
       Z1.append(z1)

                                       #TODO X_Ist
       result1 = report.results["PT1Regler.x_Ist"]
       x_Ist = result1.pretty_string()
       x_V = result1.values
       x_T = result1.time
                                    # TODO Schwingweite X_Ist
       Ueberschwing2 = max(result1.values)
       Schwingweite2 = max(result1.values) + abs(min(result1.values))
       Ueberschwing_X.append(Ueberschwing2)
       Schwing_X.append(Schwingweite2)
                                       #TODO Energie von X_Ist
       f_X = InterpolatedUnivariateSpline(x_T, x_V, k=1)
       Energie_X = scipy.integrate.quad_vec(lambda t: np.absolute(f_X(t)), 0, x_T[-1])
       E_X.append(Energie_X)
                                        # TODO Zero Crossing

       z2 = timeseries.zero_crossings(result1, 1e-9)
       Z2.append(z2)

    print(Kr_List)
    print("Zero Crossings_Alpha:",Z1)
    print("Zero Crossings_Xist:", Z2)
    print("Uberschwingweite von Alpha:", Ueberschwing_Alpha)
    print("Uberschwingweite von x_Ist:", Ueberschwing_X)
    print("Schwingweite von Alpha:", Schwing_Alpha)
    print("Schwingweite von x_Ist:", Schwing_X)
    print("Energei von Alpha:", E_Alpha)
    print("Energei von x_Ist:", E_X)
    K1 = Kr_List[Ueberschwing_Alpha.index(min(Ueberschwing_Alpha))]
    K2 = Kr_List[Ueberschwing_X.index(min(Ueberschwing_X))]
    K3 = Kr_List[E_Alpha.index(min(E_Alpha))]
    K4 = Kr_List[E_X.index(min(E_X))]
    K5 = Kr_List[Z1.index(min(Z1))]
    K6 = Kr_List[Z2.index(min(Z2))]
    Ki_List = [float(K1), float(K2), float(K3), float(K4), float(K5), float(K6)]
    print(Ki_List)
    d = defaultdict(float)
    for i in Ki_List:
        d[i] += 1
    most_frequent = sorted(Counter(Ki_List).most_common())[0]
    print(most_frequent)





    # counter = 0
    # num = Ki_List[0]
    # for i in Ki_List:
    #     curr_frequncy = Ki_List.count(i)
    #     if (curr_frequency > counter):
    #         counter = curr_frequncy
    #         num = i
    # return num




    # def test_experiment():
    #     info = loader.load_test("test.yml")
    #     details = simulink.load_details("matlab_detail.yml")
    #     details.devices[0].data["params"]["D"][1] = 0.2
    #     Kr = details.devices[0].data["params"]["Kr"]
    #     I = [r * 0.001 for r in range(0, 30, 5)]
    #     Kr_List = []
    #     Ueberschwing_Alpha = []
    #     Ueberschwing_X = []
    #     Schwing_Alpha = []
    #     Schwing_X = []
    #     E_Alpha = []
    #     E_X = []
    #     Z1 = []
    #     Z2 = []
    #     for i in I:
    #         details.devices[0].data["params"]["D"][1] = 0.2+i
    #         experiment = simulink.create(info, details)
    #         report = experiment.execute()
    #         if report.failed:
    #             raise AssertionError(report.what)
    #         print("Td:", float(0.45 + i))
    #         Kr_List.append(float(0.45 + i))
    #         # TODO Alpha
    #         result = report.results["PT1Regler.Alpha"]
    #         alpha = result.pretty_string()
    #         alpha_V = result.values
    #         alpha_T = result.time
    #         # TODO Schwingweite Alpha
    #         Ueberschwing1 = max(alpha_V)
    #         Schwingweite1 = max(result.values) + abs(min(result.values))
    #         Ueberschwing_Alpha.append(Ueberschwing1)
    #         Schwing_Alpha.append(Schwingweite1)
    #         # TODO Energie von Alpha
    #         f_Alpha = InterpolatedUnivariateSpline(alpha_T, alpha_V, k=1)
    #         Energie_Alpha = scipy.integrate.quad_vec(lambda t: np.absolute(f_Alpha(t)), 0, alpha_T[-1])
    #         E_Alpha.append(Energie_Alpha)
    #         # TODO Zero Crossing
    #
    #         z1 = timeseries.zero_crossings(result, 1e-9)
    #         Z1.append(z1)
    #
    #         # TODO X_Ist
    #         result1 = report.results["PT1Regler.x_Ist"]
    #         x_Ist = result1.pretty_string()
    #         x_V = result1.values
    #         x_T = result1.time
    #         # TODO Schwingweite X_Ist
    #         Ueberschwing2 = (resulmaxt1.values)
    #         Schwingweite2 = max(result1.values) + abs(min(result1.values))
    #         Ueberschwing_X.append(Ueberschwing2)
    #         Schwing_X.append(Schwingweite2)
    #         # TODO Energie von X_Ist
    #         f_X = InterpolatedUnivariateSpline(x_T, x_V, k=1)
    #         Energie_X = scipy.integrate.quad_vec(lambda t: np.absolute(f_X(t)), 0, x_T[-1])
    #         E_X.append(Energie_X)
    #         # TODO Zero Crossing
    #
    #         z2 = timeseries.zero_crossings(result1, 1e-9)
    #         Z2.append(z2)

       # T = result.time
       # print(T[G[0]])
       # f = InterpolatedUnivariateSpline(T, a, k=1)
       # f_int = scipy.integrate.quad_vec(lambda t: np.absolute(f(t)), 0, T[-1])
       # print(f_int)
       # B = []
       # for i in a:
       #     if abs(i)<1e-9:
       #         continue
       #         B.append(i)
       # return B
       # print(B)
    # y = f(T)
    # print(y)
    # abs_Y = abs(f(T))
    # print(abs_Y)
    # y_int = integrate.cumtrapz(abs_Y, T, initial=0)
    # qq = [Y.integral(0, t) for t in T]
    # print(qq)
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
    # a = result.values
    # asign = np.sign(a)
    # signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    # A = signchange
    # G = [np.where(A == 1)]
    # G.append(G)