import pytest

from scikit_package.utils.pypi import check_pypi_package_exists


def test_check_pypi_package_exists(mocker, capsys):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"info": {"version": "1.2.3"}}
    mock_get = mocker.patch(
        "scikit_package.utils.pypi.requests.get", return_value=mock_response
    )
    check_pypi_package_exists("my-package")
    captured = capsys.readouterr()
    assert (
        "> my-package is available on PyPI (latest version: 1.2.3)."
        in captured.out
    )
    mock_get.assert_called_once_with("https://pypi.org/pypi/my-package/json")


def test_check_pypi_package_exists_404(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 404
    mock_get = mocker.patch(
        "scikit_package.utils.pypi.requests.get", return_value=mock_response
    )
    with pytest.raises(
        ValueError,
        match="my-package is not found on PyPI. "
        "`package create conda-forge` currently only supports pulling "
        "information about the package from PyPI. Please ensure your "
        "package is uploaded to PyPI. If you want to upload package "
        "to conda-forge sourced from GitHub instead, "
        "please update the conda-forge recipe by hand.",
    ):
        check_pypi_package_exists("my-package")
    mock_get.assert_called_once_with("https://pypi.org/pypi/my-package/json")
