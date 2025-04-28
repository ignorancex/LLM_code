# from errors import InvalidShapeError
# from RingSystem import RingSystem
# from Ring import Ring

# from matplotlib.patches import Patch
# import matplotlib.pyplot as plt
# import numpy as np
# import logging
# import pytest

# @pytest.fixture
# def ring_system() -> RingSystem:
#     """generates a ring system for use in tests"""
#     radius_planet = 0.5
#     inner_radii = np.array([1., 6., 10., 20.])
#     outer_radii = np.array([6., 10., 15., 30.])
#     transmissions = np.array([0.2, 0.8, 0.3, 0.7])
#     inclination = 75
#     tilt = 37
#     rs = RingSystem(
#         radius_planet,
#         inner_radii, 
#         outer_radii, 
#         transmissions, 
#         inclination, 
#         tilt, 
#         logging.INFO
#     )
#     return rs

# def test_init(ring_system: RingSystem) -> None:
#     assert ring_system.planet_radius == 0.5
#     assert ring_system.inclination == 75
#     assert ring_system.tilt == 37
    
#     inner_radii = np.array([1., 6., 10., 20.])
#     outer_radii = np.array([6., 10., 15., 30.])
#     transmissions = np.array([0.2, 0.8, 0.3, 0.7])

#     for k, ring in enumerate(ring_system.rings):
#         assert isinstance(ring, Ring)
#         assert ring.inner_radius == inner_radii[k]
#         assert ring.outer_radius == outer_radii[k]
#         assert ring.transmission == transmissions[k]
#         assert ring.inclination == ring_system.inclination
#         assert ring.tilt == ring_system.tilt


# def test_init_invalid() -> None:
#     inner_radii = np.array([0., 2.])
#     outer_radii = np.array([1., 4.])
#     transmissions = np.ones(1)
    
#     with pytest.raises(InvalidShapeError) as ERROR:
#         RingSystem(0.5, inner_radii, outer_radii, transmissions, 83, 12)

#     assert str(ERROR.value) == "inner_radii (2,), outer_radii (2,), transmissions (1,) should all have the same shape."


# def test_get_inner_radii(ring_system: RingSystem) -> None:
#     inner_radii = ring_system.get_inner_radii()
#     assert np.all(inner_radii == np.array([1., 6., 10., 20.]))


# def test_get_outer_radii(ring_system: RingSystem) -> None:
#     outer_radii = ring_system.get_outer_radii()
#     assert np.all(outer_radii == np.array([6., 10., 15., 30.]))


# def test_get_transmissions(ring_system: RingSystem) -> None:
#     transmissions = ring_system.get_transmissions()
#     assert np.all(transmissions == np.array([0.2, 0.8, 0.3, 0.7]))


# def test_get_rings_data(ring_system: RingSystem) -> None:
#     inner_radii, outer_radii, transmissions = ring_system.get_rings_data()
    
#     assert np.all(inner_radii == np.array([1., 6., 10., 20.]))
#     assert np.all(outer_radii == np.array([6., 10., 15., 30.]))
#     assert np.all(transmissions == np.array([0.2, 0.8, 0.3, 0.7]))


# def test_get_num_rings(ring_system: RingSystem) -> None:
#     num_rings = ring_system.get_num_rings()
#     assert num_rings == 4


# def test_append_ring(ring_system: RingSystem) -> None:
#     ring = Ring(35, 40, 0.4)
#     ring_system.add_ring(4, ring)
    
#     inner_radii, outer_radii, transmissions = ring_system.get_rings_data()
    
#     assert np.all(inner_radii == np.array([1., 6., 10., 20., 35.]))
#     assert np.all(outer_radii == np.array([6., 10., 15., 30., 40.]))
#     assert np.all(transmissions == np.array([0.2, 0.8, 0.3, 0.7, 0.4]))


# def test_prepend_ring(ring_system: RingSystem) -> None:
#     ring = Ring(0.1, 0.5, 0.4)
#     ring_system.add_ring(0, ring)
    
#     inner_radii, outer_radii, transmissions = ring_system.get_rings_data()
    
#     assert np.all(inner_radii == np.array([0.1, 1., 6., 10., 20.]))
#     assert np.all(outer_radii == np.array([0.5, 6., 10., 15., 30.]))
#     assert np.all(transmissions == np.array([0.4, 0.2, 0.8, 0.3, 0.7]))


# def test_add_inbetween_ring(ring_system: RingSystem) -> None:
#     ring = Ring(16, 18, 0.4)
#     ring_system.add_ring(3, ring)
    
#     inner_radii, outer_radii, transmissions = ring_system.get_rings_data()
    
#     assert np.all(inner_radii == np.array([1., 6., 10., 16.,  20.]))
#     assert np.all(outer_radii == np.array([6., 10., 15., 18., 30.]))
#     assert np.all(transmissions == np.array([0.2, 0.8, 0.3, 0.4, 0.7]))


# def test_add_ring_inner_overlap(ring_system: RingSystem) -> None:
#     ring = Ring(14, 18, 0.4)
#     with pytest.raises(ValueError) as ERROR:
#         ring_system.add_ring(3, ring)
    
#     assert str(ERROR.value) == ("Ring must have an inner radius "
#         f"({ring.inner_radius:.2f}) > the preceding ring's outer radius "
#         "(15.00)")


# def test_add_ring_outer_overlap(ring_system: RingSystem) -> None:
#     ring = Ring(16, 22, 0.4)
#     with pytest.raises(ValueError) as ERROR:
#         ring_system.add_ring(3, ring)
    
#     assert str(ERROR.value) == ("Ring must have an outer radius "
#         f"({ring.outer_radius:.2f}) less than the succeeding ring's inner "
#         "radius (20.00)")


# def test_add_ring_both_overlap(ring_system: RingSystem) -> None:
#     ring = Ring(14, 22, 0.4)
#     with pytest.raises(ValueError) as ERROR:
#         ring_system.add_ring(3, ring)
    
#     assert str(ERROR.value) == ("Ring must have an inner radius "
#         f"({ring.inner_radius:.2f}) > the preceding ring's outer radius "
#         "(15.00)")


# def test_remove_ring(ring_system: RingSystem) -> None:
#     ring_system.remove_ring(2)

#     inner_radii, outer_radii, transmissions = ring_system.get_rings_data()
    
#     assert np.all(inner_radii == np.array([1., 6., 20.]))
#     assert np.all(outer_radii == np.array([6., 10., 30.]))
#     assert np.all(transmissions == np.array([0.2, 0.8, 0.7]))


# def test_replace_ring_valid(ring_system: RingSystem) -> None:
#     ring = Ring(12, 18, 0.4)
#     ring_system.replace_ring(2, ring)

#     inner_radii, outer_radii, transmissions = ring_system.get_rings_data()
    
#     assert np.all(inner_radii == np.array([1., 6., 12., 20.]))
#     assert np.all(outer_radii == np.array([6., 10., 18., 30.]))
#     assert np.all(transmissions == np.array([0.2, 0.8, 0.4, 0.7]))


# def test_replace_ring_inner_overlap(ring_system: RingSystem) -> None:
#     ring = Ring(8, 18, 0.4)
#     with pytest.raises(ValueError) as ERROR:
#         ring_system.replace_ring(2, ring)
    
#     assert str(ERROR.value) == ("Ring must have an inner radius "
#         f"({ring.inner_radius:.2f}) > the preceding ring's outer radius "
#         "(10.00)")


# def test_replace_ring_outer_overlap(ring_system: RingSystem) -> None:
#     ring = Ring(16, 22, 0.4)
#     with pytest.raises(ValueError) as ERROR:
#         ring_system.replace_ring(2, ring)
    
#     assert str(ERROR.value) == ("Ring must have an outer radius "
#         f"({ring.outer_radius:.2f}) less than the succeeding ring's inner "
#         "radius (20.00)")


# def test_replace_ring_both_overlap(ring_system: RingSystem) -> None:
#     ring = Ring(8, 18, 0.4)
#     with pytest.raises(ValueError) as ERROR:
#         ring_system.replace_ring(2, ring)
    
#     assert str(ERROR.value) == ("Ring must have an inner radius "
#         f"({ring.inner_radius:.2f}) > the preceding ring's outer radius "
#         "(10.00)")


# def test_split_rings_int(ring_system: RingSystem) -> None:
#     ring_system.split_rings(2)

#     inner_radii, outer_radii, transmissions = ring_system.get_rings_data()
    
#     assert np.all(inner_radii == np.array([1, 3.5, 6, 8, 10, 12.5, 20, 25]))
#     assert np.all(outer_radii == np.array([3.5, 6, 8, 10, 12.5, 15, 25, 30.]))
#     assert np.all(transmissions == np.array([0.2, 0.2, 0.8, 0.8, 0.3, 0.3, 0.7, 0.7]))


# def test_split_rings_array_valid(ring_system: RingSystem) -> None:
#     ring_system.split_rings(np.array([1, 2, 1, 4]))

#     inner_radii, outer_radii, transmissions = ring_system.get_rings_data()
    
#     assert np.all(inner_radii == np.array([1, 6, 8, 10, 20, 22.5, 25, 27.5]))
#     assert np.all(outer_radii == np.array([6, 8, 10, 15, 22.5, 25, 27.5, 30]))
#     assert np.all(transmissions == np.array([0.2, 0.8, 0.8, 0.3, 0.7, 0.7, 0.7, 0.7]))


# def test_split_rings_invalid_array(ring_system: RingSystem) -> None:
#     with pytest.raises(InvalidShapeError) as ERROR:
#         ring_system.split_rings(np.array([1, 3]))

#     assert str(ERROR.value) == "num_divisions (2,), self.rings (4,) should all have the same shape."


# def test_merge_rings_perfect(ring_system: RingSystem) -> None:
#     ring_system.split_rings(2)
#     ring_system.merge_rings()

#     inner_radii, outer_radii, transmissions = ring_system.get_rings_data()
    
#     assert np.all(inner_radii == np.array([1., 6., 10., 20.]))
#     assert np.all(outer_radii == np.array([6., 10., 15., 30.]))
#     assert np.all(transmissions == np.array([0.2, 0.8, 0.3, 0.7]))


# def test_merge_rings_tolerance(ring_system: RingSystem) -> None:
#     ring_system.split_rings(2)
#     inner_radii, outer_radii, transmissions = ring_system.get_rings_data()
#     num_rings = ring_system.get_num_rings()

#     tolerance = 1e-6
#     noise = np.random.normal(0, tolerance, len(transmissions))
    
#     noisy_ring_system = RingSystem(
#         ring_system.planet_radius,
#         inner_radii,
#         outer_radii,
#         transmissions + noise,
#         ring_system.inclination,
#         ring_system.tilt
#     )
    
#     noisy_ring_system.merge_rings()
#     num_noisy_rings = noisy_ring_system.get_num_rings()

#     assert num_rings == num_noisy_rings


# def test_merge_rings_no_merge(ring_system: RingSystem) -> None:
#     pre_inner_radii, pre_outer_radii, pre_transmissions = ring_system.get_rings_data()
#     ring_system.merge_rings()
#     post_inner_radii, post_outer_radii, post_transmissions = ring_system.get_rings_data()

#     assert np.all(pre_inner_radii == post_inner_radii)
#     assert np.all(pre_outer_radii == post_outer_radii)
#     assert np.all(pre_transmissions == post_transmissions)


# def test_get_patches(ring_system: RingSystem) -> None:
#     patches = ring_system._get_patches()
    
#     assert len(patches) == ring_system.get_num_rings() + 1
#     for patch in patches:
#         assert isinstance(patch, Patch)


# def test_plot(ring_system: RingSystem) -> None:
#     ax = ring_system.plot()
    
#     assert isinstance(ax, plt.Axes)
#     assert len(ax.patches) == ring_system.get_num_rings() + 1


# def test_repr(ring_system: RingSystem) -> None:
#     repr_string = ring_system.__repr__()

#     lines = []
#     lines.append("\n============================================"
#         "==========================")
#     lines.append("********************** RING SYSTEM INFORMATION "
#         "***********************")
#     lines.append("=============================================="
#         "========================\n")
    
#     # geometric parameters
#     lines.append("Geometric Parameters")
#     lines.append("--------------------\n")
#     planet_radius_string = (f"{ring_system.planet_radius:.2f}").rjust(7)
#     inclination_string = (f"{ring_system.inclination:.2f}").rjust(9)
#     tilt_string = (f"{ring_system.tilt:.2f}").rjust(16)
#     lines.append(f"Planet Radius: {planet_radius_string} [R*]")
#     lines.append(f"Inclination: {inclination_string} [deg]")
#     lines.append(f"Tilt: {tilt_string} [deg]")
#     lines.append("\n")

#     # gather ring information
#     ring_number = 1

#     # print ring information
#     lines.append("Ring Parameters")
#     lines.append("---------------\n")
#     lines.append("Ring #     Inner Radius     Outer Radius     "
#         "Transmission")
    
#     for ring_data in zip(*ring_system.get_rings_data()):
#         inner_radius, outer_radius, transmission = ring_data
#         ring_number_string = str(ring_number).rjust(4)
#         inner_radius_string = (f"{inner_radius:.2f}").rjust(11)
#         outer_radius_string = (f"{outer_radius:.2f}").rjust(11)
#         transmission_string = (f"{transmission:.2f}").rjust(13)
        
#         string_parameters = (ring_number_string, inner_radius_string, 
#             outer_radius_string, transmission_string)
        
#         lines.append("     ".join(string_parameters))

#         ring_number += 1

#     lines.append("")
#     lines.append("=============================================="
#         "========================")

#     expected_string = "\n".join(lines)

#     assert repr_string == expected_string


# def test_str(ring_system: RingSystem) -> None:
#     str_string = ring_system.__str__()
#     repr_string = ring_system.__repr__()
    
#     assert str_string == repr_string



# # inner_radii, outer_radii, transmissions = ring_system.get_rings_data()
    
# # assert np.all(inner_radii == np.array([1., 6., 10., 20.]))
# # assert np.all(outer_radii == np.array([6., 10., 15., 30.]))
# # assert np.all(transmissions == np.array([0.2, 0.8, 0.3, 0.7]))