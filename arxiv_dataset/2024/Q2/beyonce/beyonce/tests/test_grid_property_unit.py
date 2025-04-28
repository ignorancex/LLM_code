from beyonce.shallot.grid_units import Unit

def test_eclipse_duration_property_unit() -> None:
    parameter = Unit.ECLIPSE_DURATION
    assert parameter.property_unit == "Eclipse Duration"


def test_eclipse_duration_symbol() -> None:
    parameter = Unit.ECLIPSE_DURATION
    assert parameter.symbol == "$t_{ecl}$"


def test_eclipse_duration_str() -> None:
    parameter = Unit.ECLIPSE_DURATION
    name = parameter.__str__()
    assert name == "Eclipse Duration ($t_{ecl}$)"


def test_degree_property_unit() -> None:
    parameter = Unit.DEGREE
    assert parameter.property_unit == "Degree"
    

def test_degree_symbol() -> None:
    parameter = Unit.DEGREE
    assert parameter.symbol == "$^o$"


def test_degree_str() -> None:
    parameter = Unit.DEGREE
    name = parameter.__str__()
    assert name == "Degree ($^o$)"


def test_none_property_unit() -> None:
    parameter = Unit.NONE
    assert parameter.property_unit == "None"

def test_none_symbol() -> None:
    parameter = Unit.NONE
    assert parameter.symbol == "-"


def test_none_str() -> None:
    parameter = Unit.NONE
    name = parameter.__str__()
    assert name == "None (-)"