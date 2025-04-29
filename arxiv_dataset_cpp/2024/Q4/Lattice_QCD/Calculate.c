
/**
  * This file is part of Lattice_QCD.
  *
  * Lattice_QCD is free software: you can redistribute it and/or modify
  * it under the terms of the GNU General Public License as published by
  * the Free Software Foundation, either version 3 of the License, or
  * (at your option) any later version.
  *
  * Lattice_QCD is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  * GNU General Public License for more details.
  *
  * You should have received a copy of the GNU General Public License
  * along with Lattice_QCD.  If not, see <https://www.gnu.org/licenses/>.
  */
/**
 *  @file Calculate.c
 *  @brief the functions used in the calculation for the decay rate points
 *  @author Bradley V Smith
 *  @version 2
 */
 #include <math.h>
 #include "constant.h"

/**
 * finds the starting value for qsqr
 * @param ml: the variable getting squared
 * @return the starting value for qsqr
 */
double qsqrmin (double ml){
    return pow(ml,2);
}
/**
 *  performs the zz function
 *  @param qsqr: the current qsqr value
 *  @param tplus: the curret tplus value
 *  @return the zz value
 */
double zz(double qsqr, double tplus){
    double root1 = sqrt(tplus- qsqr);
    double root2 = sqrt(tplus - t0);
    double top = root1 - root2;
    double bottom = root1 + root2;
    double result = top/bottom;
    return result;
}
/**
 *  performs the pp function
 *  @param qsqr: the current qsqr value
 *  @param mpole: the current pole value
 *  @return the pp value
 */
double pp (double qsqr, double mpole){
    double polesqr = pow(mpole,2);
    double result = 1 - qsqr/polesqr;
    return result;
}
/**
 *  finds the nominal form factors
 *  @param qsqr: the current qsqr value
 *  @param param1: the first parameter for the form factor
 *  @param param2: the second parameter for the form factor
 *  @param param3: the third parameter for the form factor
 *  @return the nominal form factor
 */
double nomfactor (double qsqr, double param1, double param2, double param3){
    return 1/pp(qsqr, param1) * (param2 + param3 * zz(qsqr, pow(param1, 2)));
}
/**
 *  finds the higher order form factors
 *  @param qsqr: the current qsqr value
 *  @param param1: the first parameter for the form factor
 *  @param param2: the second parameter for the form factor
 *  @param param3: the third parameter for the form factor
 *  @param param4: the fourth parameter for the form factor
 *  @return the higher order form factor
 */
double highfactor (double qsqr, double param1, double param2, double param3, double param4){
      double parampow = pow(param1, 2);
      double zzresult = zz(qsqr, parampow);
      double ppresult = pp(qsqr, param1);
      double factor = 1 / ppresult * (param2 + param3 * zzresult + param4 * pow(zzresult,2));
      return factor;
}
/**
 *  finds the current splus value
 *  @param qsqr: the current qsqr value
 *  @return the splus value
 */
double splus(double qsqr){
    return pow(mlambdab + mlambdac, 2) - qsqr;
}
/**
 *  finds the current sminus value
 *  @param qsqr: the current qsqr value
 *  @return the sminus value
 */
double sminus(double qsqr){
    return pow(mlambdab - mlambdac, 2) - qsqr;
}
/**
 *  finds the decay at the given qsqr
 *  @param qsqr: the current qsqr
 *  @param factors: and array containing the current form factors
 *  @return the decay
 */
double dgammacalc(double qsqr, double factors[numfactors]){
    //printf("%lf = ", qsqr);//test
    double rootbase = sqrt(pow(mlambdab,2) + pow(pow(mlambdac,2) - qsqr,2)/pow(mlambdab,2)- 2*(pow(mlambdac,2)+qsqr));
    double root1 = sqrt(pow(mlambdab,2)-2*mlambdab*mlambdac+pow(mlambdac,2)-qsqr);
    double root2 = sqrt(pow(mlambdab+mlambdac,2)-qsqr);
    double root3 = sqrt(pow(mlambdab,2)+2*mlambdab*mlambdac+pow(mlambdac,2)-qsqr);
    double power1 = pow((1+gL)*(factors[1]*root1-factors[4]*root2),2);
    double f1 = 2*(pow(ml,2)+2*qsqr)*power1;
    double power2 = pow((1+gL)*(factors[1]*root1+factors[4]*root2),2);
    double f2 = 2 * (pow(ml,2)+2*qsqr) * power2;
    double power34 = pow(1/sqrt(qsqr)*(1+gL)*(factors[0]*(mlambdab + mlambdac)*root1 + factors[3] * (mlambdac - mlambdab) * root3),2);
    double f3 = pow(ml,2)*power34;
    double f4 = 2*qsqr * power34;
    double power5 = pow(1/sqrt(qsqr)*(1+gL)*(factors[2]*(mlambdab - mlambdac)*root2 - factors[5]*(mlambdab+mlambdac)*root1),2);
    double f5 =3*pow(ml,2)*power5;
    double power6 = pow(1/sqrt(qsqr)*(1+gL)*(factors[2]*(mlambdab - mlambdac)*root2 + factors[5]*(mlambdab+mlambdac)*root1),2);
    double f6 = 3 * pow(ml,2)*power6;
    double power78 = pow(1/sqrt(qsqr)*(1+gL)*(factors[0]*(mlambdab+mlambdac)* root1+ factors[3]*(mlambdab-mlambdac)*root2),2);
    double f7 = pow(ml,2)*power78;
    double f8 = 2 * qsqr * power78;
    double sum = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    double base = 1/(768* pow(m1,2)*pow(PI,3)* pow(qsqr,2))*pow(GF,2)*pow(pow(ml,2)-qsqr,2)*rootbase*pow(vbc,2)*sum * pow(10,15);
    return base;
}
/**
 *  finds the nominal form factors and their decay
 *  the following pattern is used for the form factors
 *  fplus - 0, fperp - 1, f0 - 2, gplus -3, gperp -4, g0 - 5, hplus - 6, hperp - 7, htildeplus - 8, htildeperp - 9
 *  @param qsqr: the current qsqr value
 *  @param param1: the array containing the parameters for the nominal form factors
 *  @return the nominal decay for the given qsqr
 */
double nominalcalc(double qsqr, double param1[set1]){
    double nominal[numfactors];
    /// Used to keep track of the parameters used
    int counter1 = 0;
    /// Keeps track of array position
    int i = 0;
    /// Finds the nominal fplus
    nominal[i] = nomfactor(qsqr, mb1minus, param1[counter1], param1[counter1 + 1]);
    counter1 += 2;
    i += 1;
    /// Finds the nominal fperp
    nominal[i] = nomfactor(qsqr, mb1minus, param1[counter1], param1[counter1 + 1]);
    counter1 += 2;
    i += 1;
    /// Finds the nominal f0
    nominal[i] = nomfactor(qsqr, mb0plus, param1[counter1], param1[counter1 + 1]);
    counter1 += 2;
    i += 1;
    /// Finds the nominal gplus
    nominal[i] = nomfactor(qsqr, mb1plus, param1[counter1], param1[counter1 + 1]);
    i += 1;
    /// Finds the nominal gperp
    nominal[i] = nomfactor(qsqr, mb1plus, param1[counter1], param1[counter1 + 2]);
    counter1 += 3;
    i += 1;
    /// Finds the nominal g0
    nominal[i] = nomfactor(qsqr, mb0minus, param1[counter1], param1[counter1 + 1]);
    counter1 += 2;
    i += 1;
    /// Finds the nominal hplus
    nominal[i] = nomfactor(qsqr, mb1minus, param1[counter1], param1[counter1 + 1]);
    counter1 += 2;
    i += 1;
    /// Finds the nominal hperp
    nominal[i] = nomfactor(qsqr, mb1minus, param1[counter1], param1[counter1 + 1]);
    counter1 += 2;
    i += 1;
    /// Finds the nominal htildeplus
    nominal[i] = nomfactor(qsqr, mb1plus, param1[counter1], param1[counter1 + 1]);
    i += 1;
    /// Finds the nominal htildeperp
    nominal[i] = nomfactor(qsqr, mb1plus, param1[counter1], param1[counter1 + 2]);
    /// finds the nominal decay
    return dgammacalc(qsqr, nominal);
}
/**
 *  finds the higher order form factors and their decay
 *  the following pattern is used for the form factors
 *  fplus - 0, fperp - 1, f0 - 2, gplus -3, gperp -4, g0 - 5, hplus - 6, hperp - 7, htildeplus - 8, htildeperp - 9
 *  @param qsqr: the current qsqr value
 *  @param param2: the array containing the parameters for the higher order form factors
 *  @return the higher order decay for the given qsqr
 */
double HOcalc(double qsqr, double param2[set2]){
    double ho[numfactors];
    /// Keeps track of parameters used
    int counter1 = 0;
    /// Keeps track of array position
    int i = 0;
    /// Finds the higher order fplus
    ho[i] = highfactor(qsqr, mb1minus, param2[counter1], param2[counter1 + 1], param2[counter1 + 2]);
    counter1 += 3;
    i += 1;
    /// Finds the higher order fperp
    ho[i] = highfactor(qsqr, mb1minus, param2[counter1], param2[counter1 + 1], param2[counter1 + 2]);
    counter1 += 3;
    i += 1;
    /// Finds the higher order f0
    ho[i] = highfactor(qsqr, mb0plus, param2[counter1], param2[counter1 + 1], param2[counter1 + 2]);
    counter1 += 3;
    i += 1;
    /// finds the higher order gplus
    ho[i] = highfactor(qsqr, mb1plus, param2[counter1], param2[counter1 + 1], param2[counter1 + 2]);
    i += 1;
    /// Finds the higher order gperp
    ho[i] = highfactor(qsqr, mb1plus, param2[counter1], param2[counter1 + 3], param2[counter1 + 4]);
    counter1 += 5;
    i += 1;
    /// Finds the higher order g0
    ho[i] = highfactor(qsqr, mb0minus, param2[counter1], param2[counter1 + 1], param2[counter1 + 2]);
    counter1 += 3;
    i += 1;
    /// Finds the higher order hplus
    ho[i] = highfactor(qsqr, mb1minus, param2[counter1], param2[counter1 + 1], param2[counter1 + 2]);
    counter1 += 3;
    i += 1;
    /// Finds the higher order hperp
    ho[i] = highfactor(qsqr, mb1minus, param2[counter1], param2[counter1 + 1], param2[counter1 + 2]);
    counter1 += 3;
    i += 1;
    /// Finds the higher order htildeplus
    ho[i] = highfactor(qsqr, mb1plus, param2[counter1], param2[counter1 + 1], param2[counter1 + 2]);
    i += 1;
    /// Finds the higher order htildeperp
    ho[i] = highfactor(qsqr, mb1plus, param2[counter1], param2[counter1 + 3], param2[counter1 + 4]);
    /// finds the higher order decay
    return dgammacalc(qsqr, ho);
}
/**
 *  finds the derivative of the decay with respect to a given parameter
 *  at a given qsqr using numeric differentiation
 *  @param qsar: the current qsqr
 *  @param *param: the parameters for the form function
 *  @param num: the parameter being differentiated
 *  @return the derivative for the parameter at the given qsqr
 */
double diffrentiate(double qsqr, double *param, int num){
    double derivative;
    /// The next value for the decay
    double pos;
    /// The last value for the decay
    double neg;
    /// For the nominal factors
    if (param_num == set1){
        *(param + num) += eps;
        pos = nominalcalc(qsqr, param);
        *(param + num) -= 2 * eps;
        neg = nominalcalc(qsqr, param);
        *(param + num) += eps;
    /// For higher order factors
    }else {
        *(param + num) += eps;
        pos = HOcalc(qsqr, param);
        *(param + num) -= 2* eps;
        neg = HOcalc(qsqr, param);
        *(param + num)+= eps;
    }
    /// numerical differentiation
    derivative = (pos - neg)/(2 * eps) ;
    return derivative;
}
/**
 *  calculates the error of the decay rate at a given qsqr
 *  @param qsqr: the current qsqr
 *  @param obs: the decay rate for the given qsqr
 *  @param param: the parameters used to find the given decay rate
 *  @param covar: the covariances of the the given parameters
 *  @return the error of the decay rate
 */
double resulterrorcalc(double qsqr, double obs, double *param, double *covar){
    /// cumulates the error
    double error = 0;
    ///finds error due to respective covariance
    double sum;
    for(int i = 0; i < param_num; i++){
        for(int j = 0; j < param_num; j++){
            sum = diffrentiate(qsqr, param, i) * *(covar+(i*param_num + j)) * diffrentiate(qsqr, param, j);
            error += sum;
        }
    }
    return sqrt(error);
}
/**
 *  finds the total error for the system
 *  @param resnom: the nominal decay rate
 *  @param resHO: the higher order decay rate
 *  @param errnom: the nominal error
 *  @param errHO:  the higher order error
 *  @return the total error for the system
 */
double totalerrorcalc(double resnom, double resHO, double errnom, double errHO){
    /// error of the system
    double systerr;
    /// part 1 of error
    double systerr1;
    /// part 2 of error
    double systerr2;
    ///finds systerr1
    if(errHO > errnom){
        systerr1 = sqrt(pow(errHO, 2) - pow(errnom, 2));
    } else {
        systerr1 = 0;
    }
    ///finds systerr2
    systerr2 = resHO - resnom;
    if(systerr2 < 0){
        systerr2 = - systerr2;
    }
    ///finds systerr
    if(systerr1 >= systerr2){
        systerr = systerr1;
    } else {
        systerr = systerr2;
    }
    return sqrt(pow(errnom, 2) + pow(systerr, 2));
}
