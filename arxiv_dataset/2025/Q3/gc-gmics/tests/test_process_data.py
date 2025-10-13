#!/usr/bin/env python3
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest

from gcgmics.process_data import (
    EvolutionData,
    Z0Data,
    median_and_spread,
    quantity_in_shells,
    return_plot_format_lists,
    return_print_format_lists,
)


# Mock universal_settings to avoid real dependencies
@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    mock_get = MagicMock()
    mock_get.side_effect = lambda key: {
        "Property dictionary": {
            "prop1": {
                "ylabel": "Label1",
                "yscale": "log",
                "ylim": (0, 10),
                "printlabel": "Prop1",
            },
            "prop2": {
                "ylabel": "Label2",
                "yscale": "linear",
                "ylim": (-1, 1),
                "printlabel": "Prop2",
            },
        },
        "z0_data_file": "test_z0.hdf5",
        "data_file_template": "test_evo_{}_{}.hdf5",
    }[key]

    # Mock Analysis dict
    mock_analysis = {"gc_mass": 1e5}

    monkeypatch.setattr("gcgmics.process_data.get", mock_get)
    monkeypatch.setattr("gcgmics.process_data.Analysis", mock_analysis)


# Tests for helper functions
def test_return_plot_format_lists():
    props = ["prop1", "prop2"]
    ylabels, yscales, ylims = return_plot_format_lists(props)

    assert ylabels == ["Label1", "Label2"]
    assert yscales == ["log", "linear"]
    assert ylims == [(0, 10), (-1, 1)]


def test_return_print_format_lists():
    props = ["prop1", "prop2"]
    labels = return_print_format_lists(props)
    assert labels == ["Prop1", "Prop2"]


def test_quantity_in_shells():
    quantity = np.array([1, 2, 3, 4])
    distances = np.array([0.5, 1.5, 2.5, 3.5])
    shell_r = np.array([0, 1, 2, 3, 4])

    result = quantity_in_shells(quantity, distances, shell_r)
    expected = np.array([1, 2, 3, 4])  # Sum in shells [0-1), [1-2), [2-3)
    np.testing.assert_array_equal(result, expected)


# Fixtures for HDF5 data
@pytest.fixture
def mock_z0_h5(tmp_path):
    file_path = tmp_path / "test_z0.hdf5"
    with h5py.File(file_path, "w") as f:
        # Create groups and datasets
        groups = [
            "All clusters",
            "Current clusters",
            "Fully disrupted",
            "Partially disrupted",
            "Field",
            "Undisrupted",
        ]
        for group in groups:
            grp = f.create_group(group)
            grp.create_dataset("Fe_H", data=np.array([-0.5, -1.0, 0.0]))
            grp.create_dataset("Distance", data=np.array([10.0, 20.0, 30.0]))
            grp.create_dataset("M_current", data=np.array([1e4, 2e4, 3e4]))
            grp.create_dataset("M_birth", data=np.array([2e4, 3e4, 4e4]))
            grp.create_dataset("M_disrupted", data=np.array([1e3, 2e3, 3e3]))
            grp.create_dataset(
                "Stellar_evolution_fraction", data=np.array([0.5, 0.6, 0.7])
            )
        f.create_dataset("R_half", data=np.array([3.0, 4.0, 5.0]))
    return file_path


@pytest.fixture
def mock_evo_h5(tmp_path):
    file_path = tmp_path / "test_evo_sim1_100000.0.hdf5"
    with h5py.File(file_path, "w") as f:
        grp = f.create_group("z evolution")
        grp.create_dataset("SFR", data=np.array([[1.0, 2.0], [3.0, 4.0]]))
        grp.create_dataset("Net_dM_GCdt", data=np.array([[0.1, 0.2], [0.3, 0.4]]))
        grp.create_dataset("M_gas,SF", data=np.array([[10.0, 20.0], [30.0, 40.0]]))
        grp.create_dataset("M_GC", data=np.array([[1e6, 2e6], [3e6, 4e6]]))
        grp.create_dataset("M_GC_r200", data=np.array([[5e5, 6e5], [7e5, 8e5]]))
        grp.create_dataset("M_200", data=np.array([[1e12, 2e12], [3e12, 4e12]]))
        grp.create_dataset("M_star", data=np.array([[1e10, 2e10], [3e10, 4e10]]))
        grp.create_dataset("NGC_model0", data=np.array([[100, 200], [300, 400]]))
        grp.create_dataset("NGC_model0_r200", data=np.array([[50, 60], [70, 80]]))
    return file_path


# Tests for Z0Data class
def test_Z0Data_process_data(mock_z0_h5, monkeypatch):
    monkeypatch.setattr(
        "gcgmics.process_data.get",
        lambda x: str(mock_z0_h5) if x == "z0_data_file" else None,
    )
    z0 = Z0Data("sim1")

    # Verify attributes are set
    assert hasattr(z0, "all_cluster_fe_h")
    assert hasattr(z0, "current_cluster_distance")
    assert hasattr(z0, "part_disrupted_m_birth")
    assert np.array_equal(z0.r_half, [3.0, 4.0, 5.0])  # From R_half dataset


@pytest.mark.parametrize(
    "kwarg_dict",
    [
        {
            "n_bins": 3,
            "outer_radius": 40.0,
            "m_gc": 2.0e5,
            "metallicity_range": [-1.0, 0.0],
        },
        {
            "outer_radius": 40.0,
            "m_gc": 2.0e5,
            "metallicity_range": [-1.0, 0.0],
        },
        {
            "n_bins": 3,
            "m_gc": 2.0e5,
            "metallicity_range": [-1.0, 0.0],
        },
        {
            "n_bins": 3,
            "outer_radius": 40.0,
            "metallicity_range": [-1.0, 0.0],
        },
        {
            "n_bins": 3,
            "outer_radius": 40.0,
            "m_gc": 2.0e5,
        },
    ],
)
def test_Z0Data_f_halo_vs_distance(mock_z0_h5, monkeypatch, kwarg_dict):
    monkeypatch.setattr(
        "gcgmics.process_data.get",
        lambda x: str(mock_z0_h5) if x == "z0_data_file" else None,
    )
    z0 = Z0Data("sim1")

    results = z0.f_halo_vs_distance(**kwarg_dict)
    assert len(results) == 5
    radius_mid, *ratios = results
    if "n_bins" in kwarg_dict:
        assert len(radius_mid) == kwarg_dict["n_bins"] - 1  # n_bins-1
        for ratio in ratios:
            assert ratio.shape == (kwarg_dict["n_bins"] - 1,)
    else:
        assert len(radius_mid) == 40 - 1  # n_bins-1
        for ratio in ratios:
            assert ratio.shape == (40 - 1,)


def test_Z0Data_reina_campos_f_halo(mock_z0_h5, monkeypatch):
    monkeypatch.setattr(
        "gcgmics.process_data.get",
        lambda x: str(mock_z0_h5) if x == "z0_data_file" else None,
    )
    z0 = Z0Data("sim1")

    results = z0.reina_campos_f_halo(outer_radius=40.0, metallicity_range=[-1.0, 0.0])
    assert len(results) == 4
    for val in results:
        assert isinstance(val, float)


def test_Z0Data_reina_campos_f_halo_metallicity_range(mock_z0_h5, monkeypatch):
    monkeypatch.setattr(
        "gcgmics.process_data.get",
        lambda x: str(mock_z0_h5) if x == "z0_data_file" else None,
    )
    z0 = Z0Data("sim1")

    results = z0.reina_campos_f_halo(outer_radius=40.0)
    assert len(results) == 4
    for val in results:
        assert isinstance(val, float)


# Tests for EvolutionData class
def test_EvolutionData_process_data(mock_evo_h5, monkeypatch):
    monkeypatch.setattr(
        "gcgmics.process_data.get",
        lambda x: (
            str(mock_evo_h5.parent) + "/test_evo_{}_{}.hdf5"
            if x == "data_file_template"
            else None
        ),
    )
    evo = EvolutionData("sim1")

    assert hasattr(evo, "SFR")
    assert hasattr(evo, "Net_dM_GCdt")
    assert evo.gc_mass == 1e5


def test_EvolutionData_make_specific_gas_dmgc_sfr(mock_evo_h5, monkeypatch):
    monkeypatch.setattr(
        "gcgmics.process_data.get",
        lambda x: (
            str(mock_evo_h5.parent) + "/test_evo_{}_{}.hdf5"
            if x == "data_file_template"
            else None
        ),
    )
    evo = EvolutionData("sim1")

    assert hasattr(evo, "SFR_MgasSF")
    assert hasattr(evo, "dMgc_dt_MgasSF")
    # Check calculation
    expected_sfr_mgas = np.array([[1.0 / 10.0, 2.0 / 20.0], [3.0 / 30.0, 4.0 / 40.0]])
    np.testing.assert_allclose(evo.SFR_MgasSF, expected_sfr_mgas)


def test_EvolutionData_make_eta(mock_evo_h5, monkeypatch):
    monkeypatch.setattr(
        "gcgmics.process_data.get",
        lambda x: (
            str(mock_evo_h5.parent) + "/test_evo_{}_{}.hdf5"
            if x == "data_file_template"
            else None
        ),
    )
    evo = EvolutionData("sim1")

    assert hasattr(evo, "eta")
    assert hasattr(evo, "eta_r200")
    # Check calculation
    expected_eta = np.array([[1e6 / 1e12, 2e6 / 2e12], [3e6 / 3e12, 4e6 / 4e12]])
    np.testing.assert_allclose(evo.eta, expected_eta)


def test_EvolutionData_constant_tn(mock_evo_h5, monkeypatch):
    monkeypatch.setattr(
        "gcgmics.process_data.get",
        lambda x: (
            str(mock_evo_h5.parent) + "/test_evo_{}_{}.hdf5"
            if x == "data_file_template"
            else None
        ),
    )
    evo = EvolutionData("sim1")

    assert hasattr(evo, "ngc_fixed_tn_20")
    expected_ngc = np.array(
        [[20 * 1e10 / 1e9, 20 * 2e10 / 1e9], [20 * 3e10 / 1e9, 20 * 4e10 / 1e9]]
    )
    np.testing.assert_allclose(evo.ngc_fixed_tn_20, expected_ngc)


def test_EvolutionData_ngc_ap_vs_r200_ratio(mock_evo_h5, monkeypatch):
    monkeypatch.setattr(
        "gcgmics.process_data.get",
        lambda x: (
            str(mock_evo_h5.parent) + "/test_evo_{}_{}.hdf5"
            if x == "data_file_template"
            else None
        ),
    )
    evo = EvolutionData("sim1")

    assert hasattr(evo, "NGC_ap_r200_ratio")
    expected_ratio = np.array([[100 / 50, 200 / 60], [300 / 70, 400 / 80]])
    np.testing.assert_allclose(evo.NGC_ap_r200_ratio, expected_ratio)


def test_median_and_spread_valid():
    data = np.array([[200, 400, 600], [900, 1000, 1100]])
    med, spread = median_and_spread(data, axis=1)
    np.testing.assert_array_equal(med, [400.0, 1000.0])
    np.testing.assert_array_equal(spread, [[264.0, 932.0], [536.0, 1068.0]])


def test_median_and_spread_no_axis():
    data = np.array([[200, 400, 600], [900, 1000, 1100]])
    med, spread = median_and_spread(data)
    np.testing.assert_array_equal(med, [400.0, 1000.0])
    np.testing.assert_array_equal(spread, [[264.0, 932.0], [536.0, 1068.0]])


def test_median_and_spread_invalid():
    data = np.array([[200, "s", 600], [900, 1000, 1100]])
    with pytest.raises(TypeError):
        median_and_spread(data, axis=1)
