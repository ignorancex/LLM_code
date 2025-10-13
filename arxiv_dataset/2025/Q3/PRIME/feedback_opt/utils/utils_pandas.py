import numpy as np
import pandas as pd


class UtilsPd:
    DEG2RAD = 2 * np.pi / 360

    @staticmethod
    def pol_to_complex(df: pd.DataFrame, absolute: str, degree: str, cmplx: str):
        assert isinstance(df, pd.DataFrame)
        assert absolute in df, f"absolute collumn {absolute} not found in provided dataframe"
        assert degree in df, f"degree collumn {degree} not found in provided dataframe"
        assert isinstance(cmplx, str)

        df[cmplx] = df[absolute].to_numpy() * np.exp(1j * df[degree].to_numpy() * UtilsPd.DEG2RAD)

    @staticmethod
    def cart_to_complex(df: pd.DataFrame, real: str, imaginary: str, cmplx: str):
        assert isinstance(df, pd.DataFrame)
        assert real in df, f"real collumn {real} not found in provided dataframe"
        assert imaginary in df, f"imaginary collumn {imaginary} not found in provided dataframe"
        assert isinstance(cmplx, str)

        df[cmplx] = df[real].to_numpy() + df[imaginary].to_numpy() * 1j

    @staticmethod
    def complex_to_pol(df: pd.DataFrame, cmplx: str, absolute: str, degree: str):
        assert isinstance(df, pd.DataFrame)
        assert cmplx in df, f"complex collumn {cmplx} not found in provided dataframe"
        assert isinstance(absolute, str)
        assert isinstance(degree, str)

        df[absolute] = np.absolute(df[cmplx].to_numpy())
        df[degree] = np.angle(df[cmplx].to_numpy()) / UtilsPd.DEG2RAD

    @staticmethod
    def complex_to_cart(df: pd.DataFrame, cmplx: str, real: str, imaginary: str):
        assert isinstance(df, pd.DataFrame)
        assert cmplx in df, f"complex collumn {cmplx} not found in provided dataframe"
        assert isinstance(real, str)
        assert isinstance(imaginary, str)

        df[real] = np.real(df[cmplx].to_numpy())
        df[imaginary] = np.imag(df[cmplx].to_numpy())
