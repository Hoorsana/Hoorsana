from __future__ import annotations

import pytest

from pylab.core import timeseries
from pylab.core import loader
from pylab.core import testing
from pylab.simulink import simulink

from pylab.simulink import _engine

import scipy
import matplotlib.pylab as plt
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.integrate
import numpy
import numpy as np
from numpy import trapz
from scipy import integrate
from collections import Counter
from collections import defaultdict


_engine.import_matlab_engine("R2021b")


def test_experiment():
    info = loader.load_test("test.yml")
    details = simulink.load_details("matlab_detail.yml")
    details.devices[0].data["params"]["D"][1] = 0.1
    I = [r * 0.01 for r in range(0, 20, 5)]
    Kr_List = []
    Ueberschwing_Alpha = []
    Ueberschwing_X = []
    Schwing_Alpha = []
    Schwing_X = []
    E_Alpha = []
    Vorzeichen_A = []
    Vorzeichen_X = []
    Zero_Estimation_A = []
    Zero_Estimation_X = []
    E_X = []
    Z1 = []
    Z2 = []
    Konvergenz_Alpha = []
    Konvergenz_X = []
    for i in I:
       details.devices[0].data["params"]["Kr"] = 0.45+i
       experiment = simulink.create(info, details)
       report = experiment.execute()
       if report.failed:
           raise AssertionError(report.what)
       print("Kr:", float(0.45+i))
       Kr_List.append(float(0.45+i))
                                       # TODO Alpha
       result = report.results["PT1Regler.Alpha"]
       alpha = result.pretty_string()
       alpha_V = result.values
       alpha_T = result.time
                                       # TODO Schwingweite Alpha
       Ueberschwing1 = max(alpha_V)
       Schwingweite1 = max(result.values)+abs(min(result.values))
       Ueberschwing_Alpha.append(Ueberschwing1)
       Schwing_Alpha.append(Schwingweite1)
                                       # TODO Energie von Alpha
       f_Alpha = InterpolatedUnivariateSpline(alpha_T, alpha_V, k=1)
       Energie_Alpha = scipy.integrate.quad_vec(lambda t: np.absolute(f_Alpha(t)), 0, alpha_T[-1])
       E_Alpha.append(Energie_Alpha)
                                      # TODO Anzahl der Nullstellen

       z1 = timeseries.zero_crossings(result, 1e-9)
       Z1.append(z1)
                                    # TODO Abschaetzung der Nullstellen
       asign = np.sign(alpha_V)
       signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
       A1 = signchange
       V1 = [np.where(A1 == 1)]
       Vorzeichen_A.append(V1)

                                      # TODO Geschwindigkeit
       Konvergenz_a = timeseries.converges_to(result, 0.0, 1e-6)
       Konvergenz_Alpha.append(Konvergenz_a)


                                       # TODO X_Ist
       result1 = report.results["PT1Regler.x_Ist"]
       x_Ist = result1.pretty_string()
       x_V = result1.values
       x_T = result1.time
                                    # TODO Schwingweite X_Ist
       Ueberschwing2 = max(result1.values)
       Schwingweite2 = max(result1.values) + abs(min(result1.values))
       Ueberschwing_X.append(Ueberschwing2)
       Schwing_X.append(Schwingweite2)
                                       # TODO Energie von X_Ist
       f_X = InterpolatedUnivariateSpline(x_T, x_V, k=1)
       Energie_X = scipy.integrate.quad_vec(lambda t: np.absolute(f_X(t)), 0, x_T[-1])
       E_X.append(Energie_X)
                                        # TODO Anzahl der Nullstellen
       z2 = timeseries.zero_crossings(result1, 1e-9)
       Z2.append(z2)
                                        # TODO Abschaetzung der Nullstellen
       asign = np.sign(x_V)
       signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
       A2 = signchange
       V2 = [np.where(A2 == 1)]
       Vorzeichen_X.append(V2)

                                        # TODO Geschwindigkeit
       Konvergenz_x = timeseries.converges_to(result1, 0.0, 1e-6)
       Konvergenz_X.append(Konvergenz_x)


    print(Kr_List)
    print("Zero Crossings_Alpha:",Z1)
    print("Zero Crossings_Xist:", Z2)
    print("Uberschwingweite von Alpha:", Ueberschwing_Alpha)
    print("Uberschwingweite von x_Ist:", Ueberschwing_X)
    print("Schwingweite von Alpha:", Schwing_Alpha)
    print("Schwingweite von x_Ist:", Schwing_X)
    print("Energei von Alpha:", E_Alpha)
    print("Energei von x_Ist:", E_X)
    print("Zeitpunkt der Konvergenz von Alpha:", Konvergenz_Alpha)
    print("Zeitpunkt der Konvergenz von X:", Konvergenz_X)

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
    Kr_Opt = most_frequent[0]


    info = loader.load_test("test.yml")
    details = simulink.load_details("matlab_detail.yml")
    details.devices[0].data["params"]["Kr"] = Kr_Opt
    I = [c * 0.02 for c in range(0, 6, 1)]
    T_List = []
    Ueberschwing_Alpha = []
    Ueberschwing_X = []
    Schwing_Alpha = []
    Schwing_X = []
    E_Alpha = []
    Vorzeichen_A = []
    Vorzeichen_X = []
    Zero_Estimation_A = []
    Zero_Estimation_X = []
    E_X = []
    Z1 = []
    Z2 = []
    Konvergenz_Alpha = []
    Konvergenz_X = []
    for i in I:
        details.devices[0].data["params"]["D"][1] = 0.05+i
        experiment = simulink.create(info, details)
        report = experiment.execute()
        if report.failed:
            raise AssertionError(report.what)
        print("Td:", float(0.05 + i))
        T_List.append(float(0.05 + i))
                                                 # TODO Alpha
        result = report.results["PT1Regler.Alpha"]
        alpha = result.pretty_string()
        alpha_V = result.values
        alpha_T = result.time
                                                 # TODO Schwingweite Alpha
        Ueberschwing1 = max(alpha_V)
        Schwingweite1 = max(result.values) + abs(min(result.values))
        Ueberschwing_Alpha.append(Ueberschwing1)
        Schwing_Alpha.append(Schwingweite1)
                                                 # TODO Energie von Alpha
        f_Alpha = InterpolatedUnivariateSpline(alpha_T, alpha_V, k=1)
        Energie_Alpha = scipy.integrate.quad_vec(lambda t: np.absolute(f_Alpha(t)), 0, alpha_T[-1])
        E_Alpha.append(Energie_Alpha)
                                                 # TODO Zero Crossing

        z1 = timeseries.zero_crossings(result, 1e-9)
        Z1.append(z1)
                                         # TODO Abschaetzung der Nullstellen
        asign = np.sign(alpha_V)
        signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
        A1 = signchange
        V1 = [np.where(A1 == 1)]
        Vorzeichen_A.append(V1)
                                         # TODO Geschwindigkeit
        Konvergenz_a = timeseries.converges_to(result, 0.0, 1e-6)
        Konvergenz_Alpha.append(Konvergenz_a)

                                                # TODO X_Ist
        result1 = report.results["PT1Regler.x_Ist"]
        x_Ist = result1.pretty_string()
        x_V = result1.values
        x_T = result1.time
                                                 # TODO Schwingweite X_Ist
        Ueberschwing2 = max(result1.values)
        Schwingweite2 = max(result1.values) + abs(min(result1.values))
        Ueberschwing_X.append(Ueberschwing2)
        Schwing_X.append(Schwingweite2)
                                                 # TODO Energie von X_Ist
        f_X = InterpolatedUnivariateSpline(x_T, x_V, k=1)
        Energie_X = scipy.integrate.quad_vec(lambda t: np.absolute(f_X(t)), 0, x_T[-1])
        E_X.append(Energie_X)
                                                 # TODO Zero Crossing

        z2 = timeseries.zero_crossings(result1, 1e-9)
        Z2.append(z2)
                                                 #TODO Abschaetzung der Nullstellen
        asign = np.sign(x_V)
        signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
        A2 = signchange
        V2 = [np.where(A2 == 1)]
        Vorzeichen_X.append(V2)

                                             # TODO Geschwindigkeit
        Konvergenz_x = timeseries.converges_to(result1, 0.0, 1e-6)
        Konvergenz_X.append(Konvergenz_x)

    print(T_List)
    print("Zero Crossings_Alpha:", Z1)
    print("Zero Crossings_Xist:", Z2)
    print("Uberschwingweite von Alpha:", Ueberschwing_Alpha)
    print("Uberschwingweite von x_Ist:", Ueberschwing_X)
    print("Schwingweite von Alpha:", Schwing_Alpha)
    print("Schwingweite von x_Ist:", Schwing_X)
    print("Energie von Alpha:", E_Alpha)
    print("Energie von x_Ist:", E_X)
    print("Zeitpunkt der Konvergenz von Alpha:", Konvergenz_Alpha)
    print("Zeitpunkt der Konvergenz von X:", Konvergenz_X)
    T1 = T_List[Ueberschwing_Alpha.index(min(Ueberschwing_Alpha))]
    T2 = T_List[Ueberschwing_X.index(min(Ueberschwing_X))]
    T3 = T_List[E_Alpha.index(min(E_Alpha))]
    T4 = T_List[E_X.index(min(E_X))]
    T5 = T_List[Z1.index(min(Z1))]
    T6 = T_List[Z2.index(min(Z2))]
    Td_List = [float(T1), float(T2), float(T3), float(T4), float(T5), float(T6)]
    print(Td_List)
    d = defaultdict(float)
    for i in Td_List:
        d[i] += 1
    most_frequent = sorted(Counter(Td_List).most_common())[0]
    print(most_frequent)
    Td_Opt = most_frequent

