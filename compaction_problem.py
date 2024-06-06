# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:01:45 2024

@author: rmportel
"""

import numpy as np
import matplotlib.pyplot as plt 
from pymoo.core.problem import ElementwiseProblem

prev_sum = 0.025

class Compaction_Problem(ElementwiseProblem):
    
    def __init__(self):
        xl = [0.25, 1e+3, 1e+3, 1e+11]
        xu = [1.00, 1e+6, 1e+6, 1e+12]

        super().__init__(n_var=4, n_obj=1, n_ieq_constr=0, xl=xl, xu=xu)
        
    ## Gruenwald-Letnikov coefficients
    def gruenwald(self, alpha, number_of_points):
        Grunwald = np.zeros(number_of_points+1)
        Grunwald[0] = 1.
        for i in range (number_of_points):
            Grunwald[i+1] = (i-alpha)/(i+1)*Grunwald[i]
        return Grunwald       

    ## A & B parameters
    def ab_parameters(self, x):
        #print(x)
        alpha = x[0]
        p = x[1]
        E0 = x[2]
        E1 = x[3]
        Delta_t = 5
        a = (p/E1)*Delta_t**(-alpha)
        b = a*(E0 + E1)
        return a, b
    
    def plot_compaction(self, soma, x, sigma_exp, sigma_num):
        global prev_sum
        
        if np.sqrt(soma) < prev_sum:
            prev_sum = soma

            fig, ax = plt.subplots()
            '''
            s = '\u03B1 = ' + "{:.2f}".format(x[0]) 
            t = '$p$ = ' + "{:.2e} Pa".format(x[1]) 
            u = '$E_{0}$ = ' + "{:.2e} Pa".format(x[2])
            v = '$E_{1}$ = ' + "{:.2e} Pa".format(x[3]) 
            w = 'e = ' + "{:.2f}".format(np.sqrt(soma)) 
            fig.text(0.95, 0.8, s, fontsize=12)
            fig.text(0.95, 0.7, t, fontsize=12)
            fig.text(0.95, 0.6, u, fontsize=12)
            fig.text(0.95, 0.5, v, fontsize=12)
            fig.text(0.95, 0.0, w, fontsize=12)'''
            ax.plot(sigma_num, label = 'Numerical')
            ax.plot(sigma_exp, label = 'Experimental')
            ax.set_ylabel(u'\u03C3 [Pa]')
            ax.set_xlabel('Points')
            ax.legend()

    def _evaluate(self, x, out, *args, **kwargs):
        # x = [alpha, p, E0, E1]
        

        sigma_exp = np.array([  
                    0.156718898044571,
                    0.139057571451465,
                    0.135606507634421,
                    0.133576470094984,
                    0.132155443817377,
                    0.131140425047659,
                    0.130328410031884,
                    0.129516395016109,
                    0.128907383754277,
                    0.128298372492446,
                    0.127689361230615,
                    0.127283353722728,
                    0.126877346214840,
                    0.126674342460896,
                    0.126268334953009,
                    0.125862327445121,
                    0.125456319937234,
                    0.125456319937234,
                    0.125050312429346,
                    0.124847308675403,
                    0.124644304921459,
                    0.124238297413571])
        sigma_exp *= 10**6
        number_of_points = len(sigma_exp)
        sigma_num = np.zeros(number_of_points)
        strainlevel = 0.494949494949495
        
        a, b = self.ab_parameters(x)
        c2 = 1+a; c1 = (x[2]+b)/c2
        gr = self.gruenwald(x[0], number_of_points)

        Step = 0
        while Step < number_of_points:
            StrainFrac = 0.0
            StressFrac = 0.0
            for j in range(1, Step+1):
                StrainFrac = StrainFrac + gr[j]*strainlevel
                StressFrac = StressFrac + gr[j]*sigma_num[Step-j]
            sigma_num[Step] = c1*strainlevel+b*StrainFrac/c2-a*StressFrac/c2
            Step = Step + 1

        # erro
        soma = 0.
        for j in range(number_of_points):
            soma += ((sigma_exp[j]-sigma_num[j])/1e+5)**2
        
        self.plot_compaction(soma, x, sigma_exp, sigma_num)
          
        out["F"] = np.sqrt(soma)
        
        return out
    