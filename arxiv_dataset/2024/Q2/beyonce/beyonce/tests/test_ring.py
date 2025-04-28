# from matplotlib.patches import Patch
# from beyonce.ring_fitter.ring_system import Ring
# import pytest

# def test_create_ring_defaults() -> None:
#     """creates ring successfully with default inclination and tilt (0, 0)"""
#     ring = Ring(0.2, 1, 0.5)

#     assert ring.inner_radius == 0.2
#     assert ring.outer_radius == 1
#     assert ring.transmission == 0.5
#     assert ring.inclination == 0
#     assert ring.tilt == 0

# def test_create_ring() -> None:
#     """creates ring succesfully"""
#     ring = Ring(0.5, 12, 0.3, 74, 37)

#     assert ring.inner_radius == 0.5
#     assert ring.outer_radius == 12
#     assert ring.transmission == 0.3
#     assert ring.inclination == 74
#     assert ring.tilt == 37

# def test_create_ring_inner_radius_zero() -> None:
#     """creates rings and sets inner radius 0 -> 1e-16"""
#     ring = Ring(0, 1, 0.5)

#     assert ring.inner_radius == 1e-16
#     assert ring.outer_radius == 1
#     assert ring.transmission == 0.5
#     assert ring.inclination == 0
#     assert ring.tilt == 0

# def test_ring_negative_inner_radius() -> None:
#     """fails because of a negative inner radius"""
#     with pytest.raises(ValueError) as ERROR:
#         Ring(-1, 1, 0.5)
    
#     assert str(ERROR.value) == ("The inner_radius argument must be greater than "
#         "0.0000")

# def test_set_ring_inner_radius() -> None:
#     """successfully sets the inner radius"""
#     ring = Ring(0.1, 1, 0.5)
#     assert ring.inner_radius == 0.1
    
#     ring.inner_radius = 0.3
#     assert ring.inner_radius == 0.3

# def test_set_ring_negative_inner_radius() -> None:
#     """fails because of a negative inner radius"""
#     ring = Ring(0, 1, 0.5)
    
#     with pytest.raises(ValueError) as ERROR:
#         ring.inner_radius = -1
    
#     assert str(ERROR.value) == ("The inner_radius argument must be greater than "
#         "0.0000")

# def test_set_ring_inner_radius_greater_than_outer_radius() -> None:
#     """fails because inner radius must be < outer radius"""
#     outer_radius = 1
#     ring = Ring(0, outer_radius, 0.5)
    
#     with pytest.raises(ValueError) as ERROR:
#         ring.inner_radius = outer_radius + 1

#     assert str(ERROR.value) == ("The inner_radius argument must be less than "
#         f"{outer_radius:.4f}")

# def test_ring_outer_radius_smaller_than_inner_radius() -> None:
#     """fails because outer radius must be > inner radius"""
#     inner_radius = 1
    
#     with pytest.raises(ValueError) as ERROR:
#         Ring(inner_radius, 0.2, 0.5)
    
#     assert str(ERROR.value) == ("The outer_radius argument must be greater than "
#         f"{inner_radius:.4f}")

# def test_set_ring_outer_radius() -> None:
#     """should set outer radius successfully"""
#     ring = Ring(0, 1, 0.5)
#     assert ring.outer_radius == 1
    
#     ring.outer_radius = 2
#     assert ring.outer_radius == 2

# def test_set_ring_outer_radius_smaller_than_inner_radius() -> None:
#     """fails because outer radius must be > inner radius"""
#     inner_radius = 0.1
#     ring = Ring(inner_radius, 0.2, 0.5)

#     with pytest.raises(ValueError) as ERROR:
#         ring.outer_radius = inner_radius - 0.1
    
#     assert str(ERROR.value) == ("The outer_radius argument must be greater than "
#         f"{inner_radius:.4f}")

# def test_ring_transmission_less_than_zero() -> None:
#     """ring transmission must be between 0 and 1"""
#     with pytest.raises(ValueError) as ERROR:
#         Ring(0, 1, -1)
    
#     assert str(ERROR.value) == ("The transmission argument must be greater than or "
#         "equal to 0.0000")

# def test_ring_transmission_greater_than_one() -> None:
#     """ring transmission must be between 0 and 1"""
#     with pytest.raises(ValueError) as ERROR:
#         Ring(0, 1, 2)
    
#     assert str(ERROR.value) == ("The transmission argument must be less than or "
#         "equal to 1.0000")

# def test_set_ring_transmission() -> None:
#     """make sure ring transmission property settings work"""
#     ring = Ring(0, 1, 0.5)
#     assert ring.transmission == 0.5

#     ring.transmission = 0.3
#     assert ring.transmission == 0.3

# def test_set_ring_transmission_less_than_zero() -> None:
#     """make sure ring transmission property settings work"""
#     ring = Ring(0, 1, 0.5)
#     assert ring.transmission == 0.5
    
#     with pytest.raises(ValueError) as ERROR:
#         ring.transmission = -1
    
#     assert str(ERROR.value) == ("The transmission argument must be greater than or "
#         "equal to 0.0000")

# def test_set_ring_transmission_greater_than_one() -> None:
#     """make sure ring transmission property settings work"""
#     ring = Ring(0, 1, 0.5)
#     assert ring.transmission == 0.5
    
#     with pytest.raises(ValueError) as ERROR:
#         ring.transmission = 2
    
#     assert str(ERROR.value) == ("The transmission argument must be less than or "
#         "equal to 1.0000")

# def test_ring_inclination_less_than_zero() -> None:
#     """ring inclination must be between 0 and 90"""
#     with pytest.raises(ValueError) as ERROR:
#         Ring(0, 1, 0.5, inclination=-1)
    
#     assert str(ERROR.value) == ("The inclination argument must be greater than or "
#         "equal to 0.0000")

# def test_ring_inclination_greater_than_ninety() -> None:
#     """ring inclination must be between 0 and 90"""
#     with pytest.raises(ValueError) as ERROR:
#         Ring(0, 1, 0.5, inclination=100)
    
#     assert str(ERROR.value) == ("The inclination argument must be less than or "
#         "equal to 90.0000")

# def test_set_ring_inclination() -> None:
#     """set the inclination successfully"""
#     ring = Ring(0, 1, 0.5, inclination=12)
#     assert ring.inclination == 12
    
#     ring.inclination = 70
#     assert ring.inclination == 70

# def test_set_ring_inclination_less_than_zero() -> None:
#     """ring inclination must be between 0 and 90"""
#     ring = Ring(0, 1, 0.5)
    
#     with pytest.raises(ValueError) as ERROR:
#         ring.inclination = -1
    
#     assert str(ERROR.value) == ("The inclination argument must be greater than or "
#         "equal to 0.0000")

# def test_set_ring_inclination_greater_than_ninety() -> None:
#     """ring inclination must be between 0 and 90"""
#     ring = Ring(0, 1, 0.5)
    
#     with pytest.raises(ValueError) as ERROR:
#         ring.inclination=100
    
#     assert str(ERROR.value) == ("The inclination argument must be less than or "
#         "equal to 90.0000")

# def test_ring_tilt_out_of_range_negative() -> None:
#     """ring tilt must be between -180 and 180"""
#     with pytest.raises(ValueError) as ERROR:
#         Ring(0, 1, 0.5, tilt=-200)
    
#     assert str(ERROR.value) == ("The tilt argument must be greater than or "
#         "equal to -180.0000")

# def test_ring_tilt_out_of_range_positive() -> None:
#     """ring tilt must be between -180 and 180"""
#     with pytest.raises(ValueError) as ERROR:
#         Ring(0, 1, 0.5, tilt=200)
    
#     assert str(ERROR.value) == ("The tilt argument must be less than or "
#         "equal to 180.0000")

# def test_set_ring_tilt() -> None:
#     """ring tilt set successfully"""
#     ring = Ring(0, 1, 0.5, tilt=15)
#     assert ring.tilt

#     ring.tilt = 60
#     assert ring.tilt == 60

# def test_set_ring_tilt_out_of_range_negative() -> None:
#     """ring tilt must be between -180 and 180"""
#     ring = Ring(0, 1, 0.5)
    
#     with pytest.raises(ValueError) as ERROR:
#         ring.tilt=-200
    
#     assert str(ERROR.value) == ("The tilt argument must be greater than or "
#         "equal to -180.0000")

# def test_set_ring_tilt_out_of_range_positive() -> None:
#     """ring tilt must be between -180 and 180"""
#     ring = Ring(0, 1, 0.5)
    
#     with pytest.raises(ValueError) as ERROR:
#         ring.tilt=200
    
#     assert str(ERROR.value) == ("The tilt argument must be less than or "
#         "equal to 180.0000")

# def test_get_patches() -> None:
#     """should return a patch object"""
#     ring = Ring(0, 1, 0.5)
#     patch = ring.get_patch()
#     assert isinstance(patch, Patch)

# def test_print() -> None:
#     """should print good information"""
#     ring = Ring(0, 1, 0.5)
#     expected = ('\n============================\n***** RING INFORMATION *****'
#         '\n============================\n\nInner Radius:   0.0000 [R*]\nOuter'
#         ' Radius:   1.0000 [R*]\nTransmission:   0.5000 [-]\nInclination:    '
#         '0.0000 [deg]\nTilt:           0.0000 [deg]\n\n======================'
#         '======')
#     given = ring.__str__()
#     assert expected == given

