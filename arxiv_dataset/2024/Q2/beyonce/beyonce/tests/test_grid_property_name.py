from beyonce.shallot.grid_names import Name


def test_disk_radius() -> None:
    parameter = Name.DISK_RADIUS
    name = parameter.__str__()
    assert name == "Disk Radius"


def test_inclination() -> None:
    parameter = Name.INCLINATION
    name = parameter.__str__()
    assert name == "Inclination"


def test_tilt() -> None:
    parameter = Name.TILT
    name = parameter.__str__()
    assert name == "Tilt"


def test_fx_map() -> None:
    parameter = Name.FX_MAP
    name = parameter.__str__()
    assert name == "Fx Map"


def test_fy_map() -> None:
    parameter = Name.FY_MAP
    name = parameter.__str__()
    assert name == "Fy Map"


def test_diagnostic_map() -> None:
    parameter = Name.DIAGNOSTIC_MAP
    name = parameter.__str__()
    assert name == "Diagnostic Map"


def test_gradient() -> None:
    parameter = Name.GRADIENT
    name = parameter.__str__()
    assert name == "Gradient"


def test_gradient_fit() -> None:
    parameter = Name.GRADIENT_FIT
    name = parameter.__str__()
    assert name == "Gradient Fit"