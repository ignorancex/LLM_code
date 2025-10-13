from abc import ABC, abstractmethod
from decimal import Decimal, getcontext
import re

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from tabulate import tabulate

from scalinglaw_utils.scaling_law_fiting.untils_decimal import NumpyMathUtil, DecimalMathUtil


class NonLinearFit(ABC):
    @abstractmethod
    def __init__(self, params_dict:dict, decimal_mode:bool):
        pass

    @abstractmethod
    def __call__(self, X: ndarray) -> ndarray|list[Decimal]:
        pass

    @abstractmethod
    def num_params(self) -> int:
        pass

    @abstractmethod
    def set_params(self, params: np.ndarray):
        pass

    def get_default_params(self) -> np.ndarray:
        pass

    @abstractmethod
    def batch_pred(self, df: DataFrame, N_col: str, D_col: str, output_col: str, is_copy:bool) -> DataFrame:
        pass

    @abstractmethod
    def predict(self, N: float, D: float) -> ndarray|Decimal:
        pass

    @abstractmethod
    def array_pred(self, df: DataFrame, N_col: str, D_col: str):
        pass

    def pred_ndarray(self, N:np.ndarray, D:np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def pred_with_given_N(self, N:float, D:ndarray) -> ndarray:
        pass


class ExpPowPowBNlogFit(NonLinearFit):
    latex_template = r"{E_show} + \exp\left({s_show} N^{{{q}}} {S_show}\right) + \left({B_show} N^{{{b}}} {Q_show}\right) D^{{{A} N^{{{a}}}}}"

    def __init__(
            self,
            params_dict: dict = None,
            decimal_mode: bool = False,
            decimal_prec: int = 30
    ):
        self.decimal_mode = decimal_mode
        self.math_util = self._init_math_util(decimal_prec)
        self.params = self._init_params(params_dict)
        if self.params is not None:
            self.set_params_from_dict(self.params)

    def _init_math_util(self, decimal_prec: int):
        if self.decimal_mode:
            getcontext().prec = decimal_prec
            getcontext().Emax = 10**7
            getcontext().Emin = -10**7

            return DecimalMathUtil()
        else:
            return NumpyMathUtil()

    def _init_params(self, params_dict: dict) -> dict:
        if params_dict is None:
            return {}

        params = {}
        for key in ["E", "s", "q", "S", "B", "b", "Q", "A", "a"]:
            if key not in params_dict:
                raise ValueError(f"Missing parameter: {key}")
            value = params_dict[key]
            if self.decimal_mode:
                params[key] = Decimal(str(value))
            else:
                params[key] = np.float64(value)
        return params

    def __call__(self, X: ndarray|list[Decimal]) -> ndarray|list[Decimal]:
        if self.decimal_mode:
            return self._call_decimal(X)
        else:
            return self._call_float(X)

    def _call_float(self, X: ndarray) -> ndarray:
        X = np.asarray(X, dtype=np.float64)
        N = X[:, 0]
        D = X[:, 1]
        efn = self.pred_fn(N)
        bn = self.pred_bn(N)
        an = self.pred_an(N)
        return efn + self.pred_hnd(D, an, bn)

    def _call_decimal(self, X: ndarray) -> list[Decimal]:
        X_decimal = [[Decimal(str(x)) for x in row] for row in X]
        results = []
        for n, d in X_decimal:
            efn = self.pred_fn(n)
            bn = self.pred_bn(n)
            an = self.pred_an(n)
            hnd = self.pred_hnd(d, an, bn)
            results.append(efn + hnd)
        return results

    def pred_fn(self, N:ndarray|Decimal) -> ndarray|Decimal:
        return self.math_util.exp(self.S + self.s * (N ** self.q))


    def pred_bn(self, N:ndarray|Decimal) -> np.ndarray|Decimal:
        return self.B * (N ** self.b) + self.Q


    def pred_an(self, N:ndarray|Decimal) -> np.ndarray|Decimal:
        return -(self.A * (N ** self.a) + self.E)
    

    def pred_hnd(self, D:ndarray|Decimal, AN:ndarray|Decimal, BN:ndarray|Decimal) -> np.ndarray|Decimal:
        d_factor = (10 ** (AN * self.math_util.log10(D)))
        return BN * d_factor

    def __str__(self):
        formatted = {}
        for param in ["E", "s", "q", "S", "B", "b", "Q", "A", "a"]:
            value = self.params[param]
            sign, abs_str = self._format_param(value)
            formatted[f"{param}_show"] = f"{sign}{abs_str}"
            formatted[param] = abs_str
        format_num = self.latex_template.format(**formatted)
        formatted = {}
        for param in ["E", "s", "q", "S", "B", "b", "Q", "A", "a"]:
            formatted[f"{param}_show"] = param
        format_note = self.latex_template.format(**formatted)

        processed_latex = re.sub(
            r'((?:[+-]\s*)+)',
            lambda m: self._reduce_signs(m.group(1)),
            format_note + "\\\\" + format_num
        )
        return processed_latex

    def _reduce_signs(self, sign_group: str) -> str:
        signs = sign_group.replace(" ", "")
        minus_count = signs.count('-')
        return '-' if minus_count % 2 == 1 else '+'

    def _format_param(self, value) -> tuple:
        if isinstance(value, Decimal):
            value = float(value)
        abs_val = abs(value)
        if abs_val == 0:
            return ('', '0')

        sign = '+' if value >= 0 else '-'

        if 1e-3 <= abs_val < 1e4:
            abs_str = f"{abs_val:.4f}".rstrip('0').rstrip('.')
        else:
            abs_str = f"{abs_val:.4e}".replace("e-0", "e-").replace("e+0", "e+")
        return (sign, abs_str)


    def num_params(self):
        return 9

    def set_params(self, params: np.ndarray):
        if len(params) != 9:
            raise ValueError("DefaultModel 需要9个参数.")
        (self.E, self.s, self.q, self.S,
         self.B, self.b, self.Q, self.A, self.a) = params

    def set_params_from_dict(self, params_dict: dict):
        required_keys = ["E", "s", "q", "S", "B", "b", "Q", "A", "a"]
        if not all(key in params_dict for key in required_keys):
            raise ValueError("参数字典中缺少必要的参数。")

        for key, value in params_dict.items():
            setattr(self, key, value)

    def batch_pred(self, df:DataFrame, N_col: str, D_col: str, output_col: str, is_copy=True) -> DataFrame:
        predictions = self.array_pred(df, N_col, D_col)
        df_c = df.copy() if is_copy else df
        df_c[output_col] = predictions
        return df_c

    def array_pred(self, df:DataFrame, N_col: str, D_col: str):
        N = df[N_col].values
        D = df[D_col].values
        X = np.column_stack((N, D))
        predictions = self(X)
        return predictions

    def predict(self, N: float, D: float):
        X = np.array([[N, D]])
        return self(X)


    def pred_ndarray(self, N:ndarray, D:ndarray) -> ndarray:
        X = np.column_stack((N, D))
        return self(X)

    def pred_with_given_N(self, N: float, D: ndarray)->ndarray:
        N_array = np.full_like(D, N)
        X = np.column_stack((N_array, D))
        return self(X)

    def plot_3d(self, N_range=(1e9, 70e9), D_range=(20e9, 500e9), n_points=50):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        ax = plt.axes(projection='3d')

        N = np.linspace(*N_range, n_points)
        D = np.linspace(*D_range, n_points)
        N_grid, D_grid = np.meshgrid(N, D)

        X = np.column_stack((N_grid.ravel(), D_grid.ravel()))
        Z = self(X).reshape(N_grid.shape)

        surf = ax.plot_surface(N_grid, D_grid, Z, cmap='viridis', alpha=0.8)

        ax.set_xlabel('N', fontsize=12)
        ax.set_ylabel('D', fontsize=12)
        ax.set_zlabel('L(N,D)', fontsize=12)
        plt.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        min_idx = np.unravel_index(np.argmin(Z, axis=None), Z.shape)
        print(f"Min value at N={N[min_idx[0]]:.2e}, D={D[min_idx[1]]:.2e}, L={Z[min_idx]:.2e}")

    
    def plot_dL_dD_dN(self, N_range=(1e9, 70e9), D_range=(20e9, 500e9), n_points=50):
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D

        N_vals = np.linspace(*N_range, n_points)
        D_vals = np.linspace(*D_range, n_points)
        N_grid, D_grid = np.meshgrid(N_vals, D_vals)

        dL_dN_values = self.dL_dN(N_grid, D_grid)
        dL_dD_values = self.dL_dD(N_grid, D_grid)

        fig1 = plt.figure(figsize=(14, 6))
        fig1.suptitle('∂L/∂N Visualization', fontsize=16)

        ax1_3d = fig1.add_subplot(1, 2, 1, projection='3d')
        surf1 = ax1_3d.plot_surface(N_grid, D_grid, dL_dN_values, cmap='viridis', alpha=0.8)
        ax1_3d.set_xlabel('N', fontsize=12)
        ax1_3d.set_ylabel('D', fontsize=12)
        ax1_3d.set_zlabel('∂L/∂N', fontsize=12)
        ax1_3d.set_title('3D Surface Plot', fontsize=14)
        fig1.colorbar(surf1, shrink=0.5, aspect=5, ax=ax1_3d, label='∂L/∂N')

        ax1_heatmap = fig1.add_subplot(1, 2, 2)
        im1 = ax1_heatmap.imshow(dL_dN_values, extent=[N_range[0], N_range[1], D_range[0], D_range[1]],
                                 origin='lower', aspect='auto', cmap='viridis')
        ax1_heatmap.set_xlabel('N', fontsize=12)
        ax1_heatmap.set_ylabel('D', fontsize=12)
        ax1_heatmap.set_title('Heatmap', fontsize=14)
        fig1.colorbar(im1, ax=ax1_heatmap, label='∂L/∂N')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        fig2 = plt.figure(figsize=(14, 6))
        fig2.suptitle('∂L/∂D Visualization', fontsize=16)

        ax2_3d = fig2.add_subplot(1, 2, 1, projection='3d')
        surf2 = ax2_3d.plot_surface(N_grid, D_grid, dL_dD_values, cmap='plasma', alpha=0.8)
        ax2_3d.set_xlabel('N', fontsize=12)
        ax2_3d.set_ylabel('D', fontsize=12)
        ax2_3d.set_zlabel('∂L/∂D', fontsize=12)
        ax2_3d.set_title('3D Surface Plot', fontsize=14)
        fig2.colorbar(surf2, shrink=0.5, aspect=5, ax=ax2_3d, label='∂L/∂D')

        ax2_heatmap = fig2.add_subplot(1, 2, 2)
        im2 = ax2_heatmap.imshow(dL_dD_values, extent=[N_range[0], N_range[1], D_range[0], D_range[1]],
                                 origin='lower', aspect='auto', cmap='plasma')
        ax2_heatmap.set_xlabel('N', fontsize=12)
        ax2_heatmap.set_ylabel('D', fontsize=12)
        ax2_heatmap.set_title('Heatmap', fontsize=14)
        fig2.colorbar(im2, ax=ax2_heatmap, label='∂L/∂D')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        print("\n部分点位的偏导数值:")
        indices_to_print = [
            (n_points // 4, n_points // 4),
            (n_points // 4, 3 * n_points // 4),
            (n_points // 2, n_points // 2),
            (3 * n_points // 4, n_points // 4),
            (3 * n_points // 4, 3 * n_points // 4),
        ]

        for r_idx, c_idx in indices_to_print:
            N_val = N_grid[r_idx, c_idx]
            D_val = D_grid[r_idx, c_idx]
            dLdN_val = dL_dN_values[r_idx, c_idx]
            dLdD_val = dL_dD_values[r_idx, c_idx]
            print(f"  索引(行={r_idx}, 列={c_idx}) -> N={N_val:.3e}, D={D_val:.3e}:")
            print(f"    ∂L/∂N = {dLdN_val:.4e}")
            print(f"    ∂L/∂D = {dLdD_val:.4e}")


        plt.show()

        

    def find_optimal_nd(self, ND_fixed: float, N_bounds=(1e9, 3e11), n_points=2000, if_plot=False):
        N_values = np.logspace(np.log10(N_bounds[0]), np.log10(N_bounds[1]), n_points)

        D_values = ND_fixed / N_values

        losses = np.array([self.predict(N, D) for N, D in zip(N_values, D_values)])

        min_loss_idx = np.argmin(losses)
        optimal_N = N_values[min_loss_idx].item()
        optimal_D = D_values[min_loss_idx].item()
        min_loss = losses[min_loss_idx].item()

        if if_plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(N_values, losses, label='Loss')
            plt.axvline(x=optimal_N, color='r', linestyle='--', label=f'Optimal N = {optimal_N:.2e}')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('N (log scale)', fontsize=12)
            plt.ylabel('Loss (log scale)', fontsize=12)
            plt.title(f'N vs Loss (ND_fixed = {ND_fixed:.2e})', fontsize=14)
            plt.legend()
            plt.grid(True, which="both", ls="--")
            plt.show()

            print(
                f"Optimal N: {optimal_N:.2e}, Optimal D: {optimal_D:.2e}, Min Loss: {min_loss:.2e}, Optimal D/N: {optimal_D / optimal_N}")
        return optimal_N, optimal_D, min_loss
    
    def find_optimal_nd_gradient(self, ND_fixed: float, N_start=None, max_steps=1000, if_plot=False):
        if N_start is None:
            N_start = np.sqrt(ND_fixed/20000)
        
        N_current = N_start
        D_current = ND_fixed / N_current
        loss_current = self.predict(N_current, D_current)
        
        N_history = [N_current]
        loss_history = [loss_current]
        
        step_size = N_current * 0.01
        
        for step in range(max_steps):
            N_new = N_current + step_size
            D_new = ND_fixed / N_new
            loss_new = self.predict(N_new, D_new)
            
            if loss_new > loss_current:
                step_size = step_size * 0.5
                if step_size < N_current * 0.0001:
                    break
                continue
            
            N_current = N_new
            D_current = D_new
            loss_current = loss_new
            
            N_history.append(N_current)
            loss_history.append(loss_current)
        
        optimal_N = N_current
        optimal_D = ND_fixed / optimal_N
        min_loss = loss_current
        
        if if_plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(N_history, loss_history, 'o-')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('N (log scale)')
            plt.ylabel('Loss (log scale)')
            plt.title(f'Optimization Process (ND_fixed = {ND_fixed:.2e})')
            plt.grid(True, which="both", ls="--")
            plt.show()
            
            print(f"Optimal N: {optimal_N:.2e}, Optimal D: {optimal_D:.2e}, Min Loss: {min_loss:.2e}, Optimal D/N: {optimal_D / optimal_N}")
        
        return optimal_N, optimal_D, min_loss


class PowPowPowBNlogFit(ExpPowPowBNlogFit):
    def pred_fn(self, N:ndarray|Decimal) -> ndarray|Decimal:
        return self.S + self.s * (N **  self.q)
    
class PowPowPowFit(ExpPowPowBNlogFit):
    def pred_fn(self, N:ndarray|Decimal) -> ndarray|Decimal:
        return self.S + self.s * (N **  self.q)

class ExpPowPowFit(ExpPowPowBNlogFit):
    def pred_bn(self, N:ndarray|Decimal) -> ndarray|Decimal:
        return self.B * (N ** self.b) + self.Q

class ExpExpPowFit(ExpPowPowFit):
    latex_template = r"\exp\left({s_show} N^{{{q_show}}} + {S_show}\right) + \exp\left({B_show} N^{{{b_show}}}+ {Q_show}\right) D^{{{A_show} N^{{{a_show}}}}}"

    def pred_bn(self, N:ndarray|Decimal) -> ndarray|Decimal:
        factor_N = self.math_util.log10(N) * self.b
        factor = (10 ** factor_N) * self.B + self.Q
        return self.math_util.exp(factor)

class PowExpPowFit(ExpExpPowFit):
    latex_template = r"( {q_show} + {S_show} N^{{{s_show}}}) + \exp({B_show} N^{{{b_show}}} + {Q_show}) D^{{{A_show} N^{{{a_show}}}}}"

    def pred_fn(self, N:ndarray|Decimal) -> ndarray|Decimal:
        factor_N = self.math_util.log10(N) * self.q
        return (10 ** factor_N) * self.s + self.S

class ExpExpExpFit(ExpExpPowFit):
    latex_template = r"\exp\left({s_show} N^{{{q_show}}} + {S_show}\right) + \exp\left({B_show} N^{{{b_show}}} + {Q_show}\right) D^{{-\exp\left({A_show} N^{{{a_show}}}+{E_show}\right)}}"

    def pred_an(self, N:ndarray|Decimal) -> ndarray|Decimal:
        factor_N = self.a * self.math_util.log10(N)
        factor = self.A * (10 ** factor_N) + self.E
        return -self.math_util.exp(factor)

    def dL_dN(self, N, D):
        exp = self.math_util.exp
        pow = np.pow
        log = self.math_util.log

        fn_N = exp(self.s * pow(N, self.q) + self.S)
        bn_N = exp(self.B * pow(N, self.b) + self.Q)
        exp_an_part = exp(self.E + self.A * pow(N, self.a))
        an_N = -exp_an_part

        term1_deriv = fn_N * (self.s * self.q * pow(N, self.q - 1))

        D_pow_an = pow(D, an_N)
        term2_bracket = (self.B * self.b * pow(N, self.b - 1)
                         - log(D) * exp_an_part * (self.A * self.a * pow(N, self.a - 1)))
        term2_deriv = bn_N * D_pow_an * term2_bracket

        return term1_deriv + term2_deriv


    def dL_dD(self, N, D):
        exp = self.math_util.exp
        pow = np.pow

        bn_N = exp(self.B * pow(N, self.b) + self.Q)
        exp_an_part = exp(self.E + self.A * pow(N, self.a))
        an_N = -exp_an_part

        D_pow_an_minus_1 = pow(D, an_N - 1)
        dL_dD = bn_N * an_N * D_pow_an_minus_1

        return dL_dD
        
        

class PowExpExpFit(ExpExpExpFit):
    latex_template = r"( {q_show} + {S_show} N^{{{s_show}}}) + \exp({B_show} N^{{{b_show}}} + {Q_show}) D^{{-\exp({A_show} N^{{{a_show}}}+{E_show})}}"

    def pred_fn(self, N:ndarray|Decimal) -> ndarray|Decimal:
        factor_N = self.math_util.log10(N) * self.q
        return (10 ** factor_N) * self.s + self.S

class PowPowExpFit(PowExpExpFit):
    def pred_bn(self, N:ndarray|Decimal) -> ndarray|Decimal:
        return self.B * (N ** self.b) + self.Q
    
class ExpPowExpFit(ExpExpExpFit):
    def pred_bn(self, N:ndarray|Decimal) -> ndarray|Decimal:
        return self.B * (N ** self.b) + self.Q

class PowConConFit(ExpPowPowBNlogFit):
    def pred_fn(self, N:ndarray|Decimal) -> ndarray|Decimal:
        return self.E + self.A * (N ** self.a)

    def pred_bn(self, N:ndarray|Decimal) -> ndarray|Decimal:
        return self.B * (N ** 0)

    def pred_an(self, N:ndarray|Decimal) -> ndarray|Decimal:
        return self.b * (N ** 0)
    
    def dL_dN(self, N, D):
        return self.A * (N ** (self.a - 1)) * self.a
    
    def dL_dD(self, N, D):
        return self.B * (D ** (self.b - 1)) * self.b 

if __name__ == "__main__":
    para_1 = {'E': 0, 'q': 0.06, 'Q': 14.200000000000001, 's': -0.5650658790499267, 'S': 1.1611030597592797,
              'B': 14.255550977435794, 'b': -1.355472407904424, 'A': -19.68057830883883, 'a': -0.20094230275346822}

    para_2 = {'E': 0.015094430185854435, 'q': -0.05701084062457085, 'Q': 0.4372270405292511, 's': 4.6671881675720215,
              'S': -2.273231029510498, 'B': 1.3253929615020752, 'b': 0.2071610987186432, 'A': -0.21445098519325256,
              'a': 0.015017773024737835}
    para_3 = {'E': 0, 'q': 0.118, 'Q': -0.2370463236411733, 's': -0.09054616402105657, 'S': 0.24706432450812132,
              'B': 474.583124812248, 'b': -0.21899999999999975, 'A': -19.68057830883883, 'a': -0.20094230275346822}

    para_6 = {'E': 0, 'q': 0.04, 'Q': -1.2222657437215778, 's': -1.3121293568002286, 'S': 2.204984751342297,
              'B': 264.41022034144504, 'b': -0.18200000000001298, 'A': -18.6532258705866, 'a': -0.19828982384906957}

    para_7 = {'E': 0, 'q': 0.034, 'Q': -1.221218470265441, 's': -1.7048291345782254, 'S': 2.649367272335889,
              'B': 255.5363422599814, 'b': -0.18030000000001317, 'A': -17.405595115941324, 'a': -0.19485496750920472}

    para_8 = {'s': 5.919935703277588, 'q': -0.0884019210934639, 'S': -1.7913458347320557, 'B': 1.3515492677688599,
              'b': -1.2697434425354004, 'Q': 4.8538923263549805, 'A': -0.43032926321029663, 'a': -0.016403961926698685,
              'E': 0.01957109570503235}

    para_9 = {'E': 0, 'q': -0.026, 'Q': -2.9083839502492106, 's': 8.019042415459078, 'S': -5.4852373891386845,
              'B': 151.17279599100195, 'b': -0.14310000000001727, 'A': -19.68057830883883, 'a': -0.20094230275346822}

    para_10 = {'E': 0, 'q': 0.023, 'Q': -1.5402725966831363, 's': -3.23738685857897, 'S': 4.412389372788829,
              'B': 231.064947761234, 'b': -0.17299999999999988, 'A': -18.6532258705866, 'a': -0.19828982384906957}
    models = [
        {"name": "use_case_lnd", "model": ExpPowPowBNlogFit(para_1)},
        {"name": "use_case_lnd_2", "model": ExpPowPowFit(para_2)},
        {"name": "use_case_lnd_3", "model": ExpExpPowFit(para_3)},
        {"name": "use_case_lnd_6", "model": ExpExpPowFit(para_6)},
        {"name": "use_case_lnd_7", "model": ExpExpPowFit(para_7)},
        {"name": "use_case_lnd_8", "model": ExpExpPowFit(para_8)},
        {"name": "use_case_lnd_9", "model": ExpExpPowFit(para_9)},
        {"name": "use_case_lnd_10", "model": ExpExpPowFit(para_10)},
    ]

    # Define the prediction points (input values) and the corresponding true values
    prediction_points = [
        (4.017E+11, 1.560E13),  # lamma-3 405B
        (6.845E+10, 2.000E+12),  # llama-2-70b
        (6.74E+09, 1.00E+12),  # llama-1-7b
        (6.369260e+09, 4.525457e+10),  # ours held out
    ]

    true_values = [
        0.3129,  # lamma-3 405B
        0.3707,  # llama-2-70b
        0.4403,  # llama-1-7b
        0.480916,  # ours held out
    ]

    # Prepare a list to store the predictions for each model and prediction point
    predictions = []

    # Loop through each model and get predictions for each point
    for model_info in models:
        row = [model_info["name"],
               model_info["model"].__class__.__name__]  # Add model name and class name from the model instance

        # Loop through each prediction point and calculate prediction and error
        for point, true_value in zip(prediction_points, true_values):
            pred = model_info["model"].predict(*point)  # Get the prediction
            pred_value = pred.item()  # Convert prediction to scalar

            # Calculate relative error
            error = (pred_value - true_value) / true_value

            # Append both predicted value and error to the row
            row.append(f"{pred_value:.5f}")  # Round to 5 decimal places
            row.append(f"{error:.5f}")  # Round error to 5 decimal places

        predictions.append(row)

    # Add the true values in a separate line for each column (Pred, error) by splitting them
    true_values_separated = ['True Value', 'Ground Truth'] # For the first two columns (Model, Model Class)
    for true_val in true_values:
        true_values_separated.append(f"{true_val:.5f}")
        true_values_separated.append('')  # Empty cell for the error column

    # Combine both the predictions and the true values row
    predictions.append(true_values_separated)

    # Define column names
    columns = ['Model', 'Model Class',
               'lamma-3 405B (Pred)', 'lamma-3 405B (error)',
               'llama-2-70b (Pred)', 'llama-2-70b (error)',
               'llama-1-7b (Pred)', 'llama-1-7b (error)',
               'ours held out (Pred)', 'ours held out (error)']

    # Create the DataFrame
    df = pd.DataFrame(predictions, columns=columns)

    # Display the table using tabulate
    print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False, floatfmt='.5f'))