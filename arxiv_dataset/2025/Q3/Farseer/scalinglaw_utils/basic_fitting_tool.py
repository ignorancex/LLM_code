
import numpy
import matplotlib.pyplot as plt
from scalinglaw_utils.architecture_search.mfa_analysis import COLORS

class BasicFit():

    def __init__(self, X, Y):
        self.x_raw = X
        self.y_raw = Y
        self.pred_y = numpy.zeros_like(Y)
        self.slope = 0
        self.intercept = 0
        self.residual = numpy.zeros_like(Y)

    def fit(self):
        pass

    def predict(self, X):
        pass

    def summary(self):
        pass

def power_law(x, a, b):
    return (a * numpy.power(x, b))

class PowerLawFit(BasicFit):

    def fit(self, method='curve_fit'):
        self.data_x = (self.x_raw / 1e+20)
        self.data_y = self.y_raw
        if (method.lower() == 'curve_fit'):
            from scipy.optimize import curve_fit
            (popt, pcov) = curve_fit(power_law, self.data_x, self.data_y, maxfev=60000)
            self.intercept = popt[0]
            self.slope = popt[1]
            self.pred_y = power_law(self.data_x, self.intercept, self.slope)
            self.pred_raw = self.pred_y
            self.residual = (self.y_raw - self.pred_raw)
            self.residual_mean = numpy.mean(numpy.absolute(self.residual))
        if (method.lower() == 'power_law'):
            from scipy.stats import powerlaw
            powerlaw.fit()
        return self

class LinearFit(BasicFit):

    def fit(self, x_log=False, y_log=False, method='scipy', weight=None, do_analysis=True):
        self.data_x = (numpy.log10(self.x_raw) if x_log else self.x_raw)
        self.data_y = (numpy.log10(self.y_raw) if y_log else self.y_raw)
        if (method.lower() == 'scipy'):
            from scipy import stats
            (self.slope, self.intercept, r_value, p_value, std_err) = stats.linregress(self.data_x.flatten(), self.data_y.flatten())
        elif (method.lower() == 'sklearn'):
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(self.data_x.reshape((- 1), 1), self.data_y.reshape((- 1), 1), weight)
            self.slope = model.coef_[0][0]
            self.intercept = model.intercept_[0]
        elif (method.lower() == 'huber'):
            from sklearn.linear_model import HuberRegressor
            huber = HuberRegressor()
            huber.fit(self.data_x.reshape((- 1), 1), self.data_y.reshape((- 1), 1), weight)
            self.slope = huber.coef_[0]
            self.intercept = huber.intercept_
        else:
            raise NotImplementedError
        if do_analysis:
            self._analysis(x_log, y_log)
        return self

    def _analysis(self, x_log, y_log):
        self.pred_y = ((self.slope * self.data_x) + self.intercept)
        self.pred_raw = (numpy.power(10, self.pred_y) if y_log else self.pred_y)
        self.residual = (self.y_raw - self.pred_raw)
        self.residual_mean = numpy.mean(numpy.absolute(self.residual))
        self.error_mean = numpy.mean(self.residual)
        self.relative_residual = (self.residual / (self.y_raw + 1e-20))
        self.relative_error_mean = numpy.mean(self.relative_residual)
        self.relative_residual_mean = numpy.mean(numpy.absolute(self.relative_residual))
        self.residual_trans = (self.data_y - self.pred_y)
        self.residual_trans_mean = numpy.mean(numpy.absolute(self.residual_trans))

    def predict(self, X):
        return ((self.slope * X) + self.intercept)

    def summary(self):
        slope_str = f'slope:{self.slope:.3e},'
        intercept_str = f'intercept:{self.intercept:.3e},'
        residual_mean_str = f'residual_mean:{self.residual_mean:.3e},'
        relative_residual_mean_str = f'relative_residual_mean:{self.relative_residual_mean:.3e},'
        residual_trans_mean_str = f'residual_trans_mean:{self.residual_trans_mean:.3e},'
        summary_str = ((((slope_str + intercept_str) + residual_mean_str) + relative_residual_mean_str) + residual_trans_mean_str)
        print(summary_str)
        return self

def plot_c19_93_test():
    c19 = numpy.array([1.241e+20, 8.1501e+19, 3.3248e+19, 2.1994e+19])
    c53 = numpy.array([3.5306e+20, 2.3177e+20, 9.4624e+19, 6.2574e+19])
    c93 = numpy.array([2.4178e+21, 6.205e+20, 4.075e+20, 1.6624e+20, 1.0997e+20])
    l19 = numpy.array([2.389, 2.433, 2.561, 2.612])
    l53 = numpy.array([2.264, 2.327, 2.423, 2.485])
    l93 = numpy.array([2.076, 2.217, 2.279, 2.382, 2.44])
    name = ['D/N=19', 'D/N=53', 'D/N=93']
    compute = [c19, c53, c93]
    loss = [l19, l53, l93]
    models = list()
    for i in range(len(compute)):
        models.append(LinearFit(compute[i], loss[i]).fit(True, True, 'sklearn'))
    plt.figure(dpi=500)
    for i in range(len(compute)):
        this_fit = models[i]
        plt.plot(this_fit.data_x, this_fit.data_y, (COLORS[i] + 'o'), label=('Data ' + name[i]))
        plt.plot(this_fit.data_x, this_fit.pred_y, (COLORS[i] + '--'), label=('Predict ' + name[i]))
        print(('The slope is %0.4f, and the intercept is %0.4f, abs_residuals is %0.4f' % (this_fit.slope, this_fit.intercept, this_fit.residual_mean)))
    plt.xlabel('Compute(Log10 Flops)')
    plt.ylabel('Smoothed Training Loss(Log10)')
    plt.legend()
    plt.grid()
    plt.show()
    plt.figure(dpi=500)
    for i in range(len(compute)):
        plt.scatter(models[i].data_x, models[i].relative_residual, color='green')
        plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Compute(Log10 Flops)')
    plt.ylabel('Training Loss')
    plt.title('Residuals')
    plt.grid(True)
    plt.show()
if (__name__ == '__main__'):
    plot_c19_93_test()
