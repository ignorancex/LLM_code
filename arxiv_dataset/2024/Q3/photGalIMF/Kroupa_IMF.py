def custom_imf(mass, time=0):
    if mass < 0.08:
        return 0
    elif mass < 0.5:
        return 2*mass**(-1.3)/4.6335594310637775
    elif mass < 1:
        return mass**(-2.3)/4.6335594310637775
    elif mass < 150 or mass == 150:
        return mass**(-2.3)/4.6335594310637775
    else:
        return 0

# from scipy.integrate import quad
#
#
# def mass_function(mass):
#     return custom_imf(mass) * mass
#
#
# integrated_mass = quad(mass_function, 0.08, 150, limit=50)[0]
# # = 4.6335594310637775
