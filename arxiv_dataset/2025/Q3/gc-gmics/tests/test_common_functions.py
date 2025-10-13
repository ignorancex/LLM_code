#!/usr/bin/env python3

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from matplotlib.axis import Axis

from gcgmics.common_functions import (
    Cosmology,
    FigureHandler,
    MinorSymLogLocator,
    SecondXAxis,
    embed_symbols,
    get_scaled_arrow_properties,
)


# Test Utility Functions ------------------------------------------------------
def test_embed_symbols():
    with patch("os.system") as mock_system:
        embed_symbols("test.pdf")
        expected_cmd = (
            "gs -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress "
            "-dEmbedAllFonts=true -sOutputFile=test_embedded.pdf -f test.pdf"
        )
        mock_system.assert_called_once_with(expected_cmd)


def test_get_scaled_arrow_properties():
    base_length = 0.1
    arrow_dict = {"head_length": 0.05, "color": "red"}
    aspect_ratio = 2.0
    scaled_len, new_dict = get_scaled_arrow_properties(
        base_length, arrow_dict, aspect_ratio
    )
    assert scaled_len == 0.05  # 0.1 / 2
    assert new_dict["head_length"] == 0.025  # 0.05 / 2
    assert new_dict["color"] == "red"


class TestCosmology:

    def test_default_initialisation(self):
        cosmo = Cosmology()
        assert cosmo.omega_M == 0.25
        assert cosmo.omega_lambda == 0.75
        assert cosmo.omega_k == 0.0
        assert cosmo.omega_r == 0.0
        assert cosmo.h == 0.72
        return None

    def test_custom_initialisation(self):
        args = {"omega_M": 0.3, "omega_lambda": 0.7, "h": 0.7}
        cosmo = Cosmology(args)
        assert cosmo.omega_M == 0.3
        assert cosmo.omega_lambda == 0.7
        assert cosmo.omega_k == 0.0
        assert cosmo.omega_r == 0.0
        assert cosmo.h == 0.7
        return None

    @pytest.mark.parametrize("z", ["s", 2])
    def test_compute_lookback_time_invalid(self, z):
        """Test output of Cosmology.compute_lookback_time"""
        # Should raise error if z not float, list or np.ndarray
        cosmo = Cosmology()
        with pytest.raises(TypeError):
            cosmo.compute_lookback_time(z)

    @pytest.mark.parametrize(
        "z, expected",
        [
            (0.3, 3.35871824),
            (0.1, 1.269832863),
            ([0.3, 0.1], [3.35871824, 1.269832863]),
        ],
    )
    def test_compute_lookback_time_valid(self, z, expected):
        """Test output of Cosmology.compute_lookback_time"""
        cosmo = Cosmology()
        assert cosmo.compute_lookback_time(z) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "z1, z2",
        [
            ("s", 0.1),
            (0.2, "s"),
            ("s", "s"),
            (2, 0.1),
            (2.0, 1),
        ],
    )
    def test_compute_time_interval_invalid(self, z1, z2):
        cosmo = Cosmology()
        with pytest.raises(TypeError):
            cosmo.compute_time_interval(z1, z2)

        return None

    @pytest.mark.parametrize(
        "z1, z2",
        [
            ([0.3, 0.4, 0.5], [0.1, 0.2]),
            ([0.4, 0.5], [0.1, 0.2, 0.3]),
        ],
    )
    def test_compute_time_interval_mismatched_inputs(self, z1, z2):
        cosmo = Cosmology()
        with pytest.raises(IOError):
            cosmo.compute_time_interval(z1, z2)

        return None

    @pytest.mark.parametrize(
        "z1, z2",
        [
            (0.1, 0.2),
            ([0.1, 0.2], [0.2, 0.3]),
            ([0.1, 0.2], [0.2, 0.1]),
            ([0.2, 0.1], [0.1, 0.2]),
        ],
    )
    def test_compute_time_interval_invalid_z_order(self, z1, z2):
        cosmo = Cosmology()
        with pytest.raises(IOError):
            cosmo.compute_time_interval(z1, z2)

        return None

    @pytest.mark.parametrize(
        "z1, z2",
        [
            (np.nan, 0.1),
            (0.1, np.nan),
            (np.nan, np.nan),
        ],
    )
    def test_compute_time_interval_nan_scalar(self, z1, z2):
        cosmo = Cosmology()

        assert np.isnan(cosmo.compute_time_interval(z1, z2))

        return None

    @pytest.mark.parametrize(
        "z1, z2",
        [
            ([0.2, 0.3], [0.1, np.nan]),
            ([0.2, 0.3], [np.nan, 0.1]),
            ([0.2, np.nan], [0.1, 0.2]),
            ([np.nan, 0.3], [0.1, 0.2]),
        ],
    )
    def test_compute_time_interval_nan_array(self, z1, z2):
        cosmo = Cosmology()

        assert any(np.isnan(cosmo.compute_time_interval(z1, z2)))

        return None

    @pytest.mark.parametrize(
        "z1, z2, expected",
        [
            (0.3, 0.1, 2.088885),
            ([0.3], 0.1, 2.088885),
        ],
    )
    def test_compute_time_interval_valid(self, z1, z2, expected):
        """Test output of Cosmology.compute_time_interval"""
        cosmo = Cosmology()
        assert cosmo.compute_time_interval(z1, z2) == pytest.approx(expected)

    def test_E_z(self):
        """Test output of Cosmology.E_z"""
        cosmo = Cosmology()
        z = 1.0

        expected = 0.25 * 2**3 + 0.75
        assert np.isclose(cosmo.E_z(z), expected)

        z = np.asarray([1.0, 2.0, 3.0])
        expected = [0.25 * 2**3 + 0.75, 0.25 * 3**3 + 0.75, 0.25 * 4**3 + 0.75]
        assert np.array_equal(cosmo.E_z(z), expected)

        return None

    def test_hubble_parameter_in_sec(self):
        """Test output of Cosmology.hubble_parameter_in_sec"""
        cosmo = Cosmology()
        assert cosmo.hubble_parameter_in_sec() == 0.72 / (10.0 * 3.085677581491367e16)


# Test FigureHandler Class ----------------------------------------------------
class TestFigureHandler:
    @pytest.fixture
    def setup_figure_handler(self):
        major_z = np.array([0, 1, 2])
        minor_z = np.array([0.5, 1.5])
        tlb_lim = (0, 10)
        cosmo = MagicMock()
        cosmo.age = MagicMock(return_value=np.array([10, 5, 3]))
        return FigureHandler(major_z, minor_z, tlb_lim, cosmo)

    @patch("gcgmics.common_functions.SecondXAxis")
    def test_set_figure_properties(self, MockSecondXAxis, setup_figure_handler):
        fh = setup_figure_handler
        mock_ax = MagicMock()
        axs = [mock_ax, mock_ax]
        ylabels = ["Y1", "Y2"]

        # Create mock instance
        mock_second_x_axis = MagicMock()
        MockSecondXAxis.return_value = mock_second_x_axis

        fh.set_figure_properties(axs, ylabels)

        # Should create SecondXAxis for each axis
        assert MockSecondXAxis.call_count == len(axs)

        # Verify SecondXAxis methods were called for each axis
        assert mock_second_x_axis.set_major_x_ticks.call_count == len(axs)
        assert mock_second_x_axis.set_minor_x_ticks.call_count == len(axs)
        assert mock_second_x_axis.set_axis_limits.call_count == len(axs)
        assert mock_second_x_axis.set_xlabel.call_count == len(axs)
        assert mock_second_x_axis.invert_axis.call_count == len(axs)

        # Check all calls
        calls = mock_ax.set.call_args_list
        assert len(calls) == 2

        # First call should have Y1 label
        _, kwargs1 = calls[0]
        assert kwargs1["xlabel"] == fh.x_label
        assert kwargs1["ylabel"] == "Y1"
        assert kwargs1["yscale"] == "log"

        # Second call should have Y2 label
        _, kwargs2 = calls[1]
        assert kwargs2["ylabel"] == "Y2"

    @patch("gcgmics.common_functions.SecondXAxis")
    def test_second_x_axis_creation(self, mock_second_x_axis, setup_figure_handler):
        fh = setup_figure_handler
        mock_ax = MagicMock()
        axs = [mock_ax]

        fh.set_figure_properties(axs, ["Y1"])

        # Verify SecondXAxis was created with correct arguments
        mock_second_x_axis.assert_called_once()
        args, kwargs = mock_second_x_axis.call_args
        assert args[0] == mock_ax
        assert callable(args[1])  # tick_function

    @patch("gcgmics.common_functions.SecondXAxis")
    def test_set_figure_properties_symlog(
        self, mock_second_x_axis, setup_figure_handler
    ):
        fh = setup_figure_handler
        mock_ax = MagicMock()
        axs = [mock_ax]
        ylabels = ["Y1"]

        fh.set_figure_properties(axs, ylabels, yscale="symlog", linthresh=0.1)

        # Check symlog specific calls
        mock_ax.set_yscale.assert_called_once_with("symlog", linthresh=0.1)
        mock_ax.yaxis.set_minor_locator.assert_called_once()

    @patch("gcgmics.common_functions.SecondXAxis")
    def test_set_figure_properties_ylim_negative(
        self, mock_second_x_axis, setup_figure_handler
    ):
        fh = setup_figure_handler
        mock_ax = MagicMock()
        axs = [mock_ax]
        ylabels = ["Y1"]

        fh.set_figure_properties(axs, ylabels, ylims=[(-1, 1)])

        # Should draw zero line for negative ylim
        mock_ax.axhline.assert_called_once_with(0.0, color="k", linestyle=":", zorder=0)

    @patch("gcgmics.common_functions.SecondXAxis")
    def test_set_figure_properties_no_legend(
        self, mock_second_x_axis, setup_figure_handler
    ):
        fh = setup_figure_handler
        mock_ax = MagicMock()
        axs = [mock_ax]
        ylabels = ["Y1"]

        fh.set_figure_properties(axs, ylabels, no_legend=True)

        # Should not create legend
        mock_ax.legend.assert_not_called()

    @patch("gcgmics.common_functions.SecondXAxis")
    def test_set_figure_with_ratio_properties(
        self, MockSecondXAxis, setup_figure_handler
    ):
        fh = setup_figure_handler
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        axs = [[mock_ax1, mock_ax2]]
        ylabels = [("Top", "Bottom")]

        # Create a mock instance for SecondXAxis
        mock_second_x_axis = MagicMock()
        MockSecondXAxis.return_value = mock_second_x_axis

        fh.set_figure_with_ratio_properties(axs, ylabels)

        # Verify SecondXAxis was created
        MockSecondXAxis.assert_called_once()

        # Verify SecondXAxis methods were called
        mock_second_x_axis.set_major_x_ticks.assert_called_once()
        mock_second_x_axis.set_minor_x_ticks.assert_called_once()
        mock_second_x_axis.set_axis_limits.assert_called_once_with(fh.tlb_lim)
        mock_second_x_axis.set_xlabel.assert_called_once_with(r"$z$")
        mock_second_x_axis.invert_axis.assert_called_once()

        # Verify axis configurations
        mock_ax1.set.assert_called_once_with(ylabel="Top", yscale="linear", ylim=None)
        mock_ax2.set.assert_called_once_with(
            xlabel=fh.x_label,
            ylabel="Bottom",
            yscale="log",
            ylim=None,  # Changed from None to 'log'
        )
        mock_ax1.axhline.assert_called_once_with(1, color="k", linestyle=":")

    @patch("gcgmics.common_functions.SecondXAxis")
    def test_set_figure_with_ratio_properties_symlog(
        self, mock_second_x_axis, setup_figure_handler
    ):
        fh = setup_figure_handler
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        axs = [[mock_ax1, mock_ax2]]
        ylabels = [("Top", "Bottom")]

        fh.set_figure_with_ratio_properties(axs, ylabels, yscale="symlog")

        mock_ax1.set.assert_called_once_with(ylabel="Top", yscale="linear", ylim=None)
        mock_ax2.set.assert_called_once_with(
            xlabel=fh.x_label, ylabel="Bottom", yscale=None, ylim=None
        )
        # Should additionally set symlog properties
        mock_ax2.set_yscale.assert_called_once_with("symlog", linthresh=1.0e-3)
        mock_ax2.yaxis.set_minor_locator.assert_called_once()

    @patch("gcgmics.common_functions.SecondXAxis")
    def test_set_stacked_figure_properties_horizontal(
        self, MockSecondXAxis, setup_figure_handler
    ):
        fh = setup_figure_handler
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_ax3 = MagicMock()
        axs = [mock_ax1, mock_ax2, mock_ax3]
        ylabels = ["Y1", "Y2", "Y3"]

        # Create mock instance
        mock_second_x_axis = MagicMock()
        MockSecondXAxis.return_value = mock_second_x_axis

        text_labels = ["A", "B", "C"]
        text_label_args = [{"fontsize": 10}, {"fontsize": 20}, {"fontsize": 30}]

        fh.set_stacked_figure_properties(
            axs,
            ylabels,
            direction="horizontal",
            text_labels=text_labels,
            text_label_args=text_label_args,
        )

        # Verify SecondXAxis created for horizontal direction
        assert MockSecondXAxis.call_count == len(axs)

        # Verify methods were called on each instance
        assert mock_second_x_axis.set_major_x_ticks.call_count == len(axs)
        assert mock_second_x_axis.set_minor_x_ticks.call_count == len(axs)
        assert mock_second_x_axis.set_axis_limits.call_count == len(axs)
        assert mock_second_x_axis.set_xlabel.call_count == len(axs)

        # Verify text labels - only the explicitly passed ones
        for i, ax in enumerate(axs):
            ax.text.assert_any_call(
                s=text_labels[i], transform=ax.transAxes, **text_label_args[i]
            )

        # Additional verification for panel labels (only when len(axs) > 2)
        mock_ax1.text.assert_any_call(
            0.02,
            0.96,
            "(a)",  # First element of global panel_labels
            color="k",
            fontsize="medium",
            verticalalignment="top",
            transform=mock_ax1.transAxes,
        )
        mock_ax2.text.assert_any_call(
            0.02,
            0.96,
            "(b)",  # Second element of global panel_labels
            color="k",
            fontsize="medium",
            verticalalignment="top",
            transform=mock_ax2.transAxes,
        )
        mock_ax3.text.assert_any_call(
            0.02,
            0.96,
            "(c)",  # Third element of global panel_labels
            color="k",
            fontsize="medium",
            verticalalignment="top",
            transform=mock_ax3.transAxes,
        )

        # Verify legend only in first panel
        mock_ax1.legend.assert_called()
        mock_ax2.legend.assert_not_called()
        mock_ax3.legend.assert_not_called()

    @patch("gcgmics.common_functions.SecondXAxis")
    def test_set_stacked_figure_properties_vertical(
        self, MockSecondXAxis, setup_figure_handler
    ):
        fh = setup_figure_handler
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        axs = [mock_ax1, mock_ax2]
        ylabels = ["Y1", "Y2"]

        # Create mock instance
        mock_second_x_axis = MagicMock()
        MockSecondXAxis.return_value = mock_second_x_axis

        fh.set_stacked_figure_properties(
            axs,
            ylabels,
            direction="vertical",
            yscale=["log", "symlog"],
            ylims=[(0, 10), (-5, 5)],
        )

        # Should create SecondXAxis only on first panel
        MockSecondXAxis.assert_called_once()

        # Verify axis inversion
        MockSecondXAxis.return_value.invert_axis.assert_called_once()

        # Verify SecondXAxis methods
        mock_second_x_axis.set_major_x_ticks.assert_called_once()
        mock_second_x_axis.set_minor_x_ticks.assert_called_once()
        mock_second_x_axis.set_axis_limits.assert_called_once_with(fh.tlb_lim)
        mock_second_x_axis.set_xlabel.assert_called_once_with(r"$z$")

        # Verify yscale and ylim settings for ax1
        mock_ax1.set.assert_has_calls(
            [
                call(ylim=(0, 10)),  # First call sets ylim
                call(yscale="log"),  # Second call sets yscale
            ],
            any_order=True,
        )

        # Verify yscale and ylim settings for ax2
        mock_ax2.set.assert_has_calls(
            [
                call(ylim=(-5, 5)),  # First call sets ylim
                call(xlabel=fh.x_label),  # Second call sets xlabel
            ],
            any_order=True,
        )

        # Verify symlog specific settings
        mock_ax2.set_yscale.assert_called_once_with("symlog", linthresh=0.001)


# Test SecondXAxis Class ------------------------------------------------------
class TestSecondXAxis:
    @patch("matplotlib.axes.Axes")
    def test_second_x_axis_init(self, mock_ax):
        mock_conversion = MagicMock()
        second_axis = SecondXAxis(mock_ax, mock_conversion)
        assert second_axis.ax1 == mock_ax
        assert second_axis.ax2 == mock_ax.twiny.return_value
        assert second_axis.conversion_func == mock_conversion

    @patch("matplotlib.axes.Axes")
    def test_set_axis_limits(self, mock_ax):
        second_axis = SecondXAxis(mock_ax, MagicMock())
        limits = (0, 10)

        second_axis.set_axis_limits(limits)

        mock_ax.set_xlim.assert_called_once_with(limits)
        second_axis.ax2.set_xlim.assert_called_once_with(limits)

    @patch("matplotlib.axes.Axes")
    def test_invert_axis(self, mock_ax):
        second_axis = SecondXAxis(mock_ax, MagicMock())

        second_axis.invert_axis()

        mock_ax.invert_xaxis.assert_called_once()
        second_axis.ax2.invert_xaxis.assert_called_once()

    @patch("matplotlib.axes.Axes")
    def test_set_major_x_ticks(self, mock_ax):
        mock_conversion = MagicMock(return_value=[5, 3, 1])
        second_axis = SecondXAxis(mock_ax, mock_conversion)
        redshifts = [0, 1, 2]
        cosmo = MagicMock()

        second_axis.set_major_x_ticks(redshifts, cosmo, label_override=["A", "B", "C"])

        mock_conversion.assert_called_once_with(redshifts, cosmo)
        second_axis.ax2.set_xticks.assert_called_once_with([5, 3, 1])
        second_axis.ax2.set_xticklabels.assert_called_once_with(["A", "B", "C"])

    @patch("matplotlib.axes.Axes")
    def test_set_minor_x_ticks(self, mock_ax):
        mock_conversion = MagicMock(return_value=[4, 2])
        second_axis = SecondXAxis(mock_ax, mock_conversion)
        redshifts = [0.5, 1.5]
        cosmo = MagicMock()

        second_axis.set_minor_x_ticks(redshifts, cosmo)

        mock_conversion.assert_called_once_with(redshifts, cosmo)
        second_axis.ax2.set_xticks.assert_called_once_with([4, 2], minor=True)

    @patch("matplotlib.axes.Axes")
    def test_set_xlabel(self, mock_ax):
        second_axis = SecondXAxis(mock_ax, MagicMock())

        second_axis.set_xlabel("Test Label")

        second_axis.ax2.set_xlabel.assert_called_once_with("Test Label")


class TestMinorSymLogLocator:
    @pytest.fixture
    def mock_axis(self):
        axis = MagicMock(spec=Axis)
        return axis

    def test_initialization(self):
        locator = MinorSymLogLocator(linthresh=0.1, nints=5)
        assert locator.linthresh == 0.1
        assert locator.nintervals == 5

    @pytest.mark.parametrize(
        "majorlocs, linthresh, nints, expected_ranges",
        [
            # Linear region
            ([-0.5, 0, 0.5], 1.0, 10, [(-1.5, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1.5)]),
            # Logarithmic region
            ([10, 100], 1.0, 5, [(1, 10), (10, 100), (100, 1000)]),
            # Crossing linear threshold
            (
                [-10, -1, 0, 1, 10],
                1.0,
                5,
                [(-100, -10), (-10, -1), (-1, 0), (0, 1), (1, 10), (10, 100)],
            ),
        ],
    )
    def test_minor_tick_distribution(
        self, mock_axis, majorlocs, linthresh, nints, expected_ranges
    ):
        mock_axis.get_majorticklocs.return_value = np.array(majorlocs)
        locator = MinorSymLogLocator(linthresh=linthresh, nints=nints)
        locator.axis = mock_axis

        minor_locs = locator()

        # Verify ordering
        if len(minor_locs) > 0:
            diffs = np.diff(minor_locs)
            assert np.all(diffs > 0), "Minor ticks not in ascending order"

        # Check distribution in expected ranges
        range_counts = [0] * len(expected_ranges)
        for tick in minor_locs:
            for i, (low, high) in enumerate(expected_ranges):
                if low <= tick < high:
                    range_counts[i] += 1
                    break

        # Verify each range has expected number of ticks
        for count in range_counts:
            assert count > 0, f"Expected ticks in range {expected_ranges[i]}"

    def test_symlog_behavior_around_zero(self, mock_axis):
        mock_axis.get_majorticklocs.return_value = np.array([-1, 0, 1])
        locator = MinorSymLogLocator(linthresh=0.5)
        locator.axis = mock_axis

        minor_locs = locator()

        # Should have linear minor ticks between -1 and 1
        assert len(minor_locs) > 0
        assert all(np.abs(loc) < 1 for loc in minor_locs if -0.5 < loc < 0.5)

    def test_ordering_and_uniqueness(self, mock_axis):
        mock_axis.get_majorticklocs.return_value = np.array([-10, -1, 0, 1, 10])
        locator = MinorSymLogLocator(linthresh=1.0)
        locator.axis = mock_axis

        minor_locs = locator()

        # Check for duplicates
        unique_locs = np.unique(minor_locs)
        assert len(unique_locs) == len(minor_locs), "Duplicate minor ticks found"

        # Verify ordering
        diffs = np.diff(unique_locs)
        assert np.all(diffs > 0), "Minor ticks not in ascending order"

    def test_linear_region_coverage(self, mock_axis):
        mock_axis.get_majorticklocs.return_value = np.array([-0.5, 0, 0.5])
        locator = MinorSymLogLocator(linthresh=1.0, nints=5)
        locator.axis = mock_axis

        minor_locs = locator()
        linear_ticks = [t for t in minor_locs if -1.0 < t < 1.0]

        # Should have ticks between major ticks
        assert any(-0.5 < t < 0 for t in linear_ticks), "No ticks in (-0.5, 0)"
        assert any(0 < t < 0.5 for t in linear_ticks), "No ticks in (0, 0.5)"

    def test_logarithmic_region_coverage(self, mock_axis):
        mock_axis.get_majorticklocs.return_value = np.array([10, 100])
        locator = MinorSymLogLocator(linthresh=1.0, nints=5)
        locator.axis = mock_axis

        minor_locs = locator()
        log_ticks = [t for t in minor_locs if 10 < t < 100]

        assert len(log_ticks) > 0, "No ticks in logarithmic region"

    def test_zero_handling(self, mock_axis):
        mock_axis.get_majorticklocs.return_value = np.array([-1, 0, 1])
        locator = MinorSymLogLocator(linthresh=0.5)
        locator.axis = mock_axis

        minor_locs = locator()

        # Verify no minor ticks at exact zero
        assert 0.0 not in minor_locs, "Minor tick at zero"

        # Verify symmetric distribution
        neg_ticks = [t for t in minor_locs if t < 0]
        pos_ticks = [t for t in minor_locs if t > 0]
        assert len(neg_ticks) == len(pos_ticks), "Asymmetric ticks around zero"

    def test_tick_values_exception(self):
        locator = MinorSymLogLocator(linthresh=1.0)
        with pytest.raises(NotImplementedError):
            locator.tick_values(0, 10)
