import numpy as np
import src.utilities.constants as const


# real life snp Progenitors live from 3 - 40 Myrs  
class Supernovae:   
	rng = np.random.default_rng() 
 
	def __init__(self, association_x, association_y, association_z, association_creation_time, simulation_time, sn_mass, one_dim_vel, expected_lifetime, vel_theta_dir, vel_phi_dir, reference_to_companion=None):
		""" Class to represent a snp progenitor. The snp progenitors are created at the same time as the association is created.
		
		Args:
			association_x (float): x-coordinate of the association. Units of kpc
			association_y (float): y-coordinate of the association. Units of kpc
			association_z (float): z-coordinate of the association. Units of kpc
			creation_time (int): how many years ago the sn/association was created. Units of Myr
			simulation_time (int): The current time of the simulation in units of Myr. In the simulation, this will decrease on each iteration, counting down from the creation_time to the present
			sn_mass (float): mass of the snp progenitor. Units of solar masses
			one_dim_vel (float): Gaussian velocity distribution with a mean of 0 km/s and a standard deviation of 2 km/s
			expected_lifetime (float): how many years it will take for the star to explode. Units of Myr
			vel_theta_dir (float): theta direction for the velocity dispersion. Units of radians
			vel_phi_dir (float): phi direction for the velocity dispersion. Units of radians

		Returns:
			None 
		"""
		
		self.__association_x = association_x
		self.__association_y = association_y
		self.__association_z = association_z
		self.__sn_x = association_x # because the SNP's are born at the association centre. Also prevents issues if the getters for SNP positions are called before calculate_position()
		self.__sn_y = association_y # because the SNP's are born at the association centre. Also prevents issues if the getters for SNP positions are called before calculate_position()
		self.__sn_z = association_z # because the SNP's are born at the association centre. Also prevents issues if the getters for SNP positions are called before calculate_position()
		self.__long = None # this is in radians! Value updated by calculate_position()
		self.__snp_creation_time = association_creation_time # how many years ago the sn/association was created
		self.__age = 0 # the age of the snp. Value updated by the setter
		self.__sn_mass = sn_mass
		self.__one_dim_vel = one_dim_vel # Gaussian velocity distribution with a mean of 0 km/s and a standard deviation of 2 km/s
		self.__expected_lifetime = expected_lifetime # Myr, how many years it will take for the star to explode
		self.__exploded = False # True if the star has exploded, False otherwise. Value dependent on creation time, age and expected_lifetime. 
		self.__vel_theta_dir = vel_theta_dir # radians
		self.__vel_phi_dir = vel_phi_dir # radians
		self.__reference_to_companion = reference_to_companion # reference to the companion star in the binary system. Value updated by the setter. None means this is a single star

	@property
	def age(self):
		if self.__age < 0:
			return # a negative age means the snp has not been created yet, but is scheduled to be created in the future. It doesn't exist - pretend like it doesn't and pass
		return self.__age

	@property
	def expected_lifetime(self):
		if self.__age < 0:
			return # a negative age means the snp has not been created yet, but is scheduled to be created in the future. It doesn't exist - pretend like it doesn't and pass
		return self.__expected_lifetime
	
	@property
	def snp_creation_time(self):
		return self.__snp_creation_time

	@property
	def velocity(self):
		if self.__age < 0:
			return # a negative age means the snp has not been created yet, but is scheduled to be created in the future. It doesn't exist - pretend like it doesn't and pass
		return self.__one_dim_vel
	
	@property
	def mass(self):
		if self.__age < 0:
			return 0 # a negative age means the snp has not been created yet, but is scheduled to be created in the future. It doesn't exist - pretend like it doesn't and return 0
		return self.__sn_mass
	
	@property
	def x(self): 
		if self.__age < 0:
			return # a negative age means the snp has not been created yet, but is scheduled to be created in the future. It doesn't exist - pretend like it doesn't and pass
		return self.__sn_x
	
	@property
	def y(self): 
		if self.__age < 0:
			return # a negative age means the snp has not been created yet, but is scheduled to be created in the future. It doesn't exist - pretend like it doesn't and pass
		return self.__sn_y
	
	@property
	def z(self): 
		if self.__age < 0:
			return # a negative age means the snp has not been created yet, but is scheduled to be created in the future. It doesn't exist - pretend like it doesn't and pass
		return self.__sn_z
	
	@property
	def vel_theta_dir(self):
		if self.__age < 0:
			return # a negative age means the snp has not been created yet, but is scheduled to be created in the future. It doesn't exist - pretend like it doesn't and pass
		return self.__vel_theta_dir
	
	@property
	def vel_phi_dir(self):
		if self.__age < 0:
			return # a negative age means the snp has not been created yet, but is scheduled to be created in the future. It doesn't exist - pretend like it doesn't and pass
		return self.__vel_phi_dir
	
	@property
	def longitude(self):
		if self.__age < 0:
			return # a negative age means the snp has not been created yet, but is scheduled to be created in the future. It doesn't exist - pretend like it doesn't and pass
		return self.__long
	
	@property
	def exploded(self):
		if self.__age < 0:
			return # a negative age means the snp has not been created yet, but is scheduled to be created in the future. It doesn't exist - pretend like it doesn't and pass
		return self.__exploded
	

	def _calculate_age(self, simulation_time):
		""" Function to calculate the age of the snp. The age is calculated at the given value for simulation_time.
		
		Args:
			simulation_time (int): The new simulation time in units of Myr. Simulation time counts down to zero from the creation time of the Galaxy.
			
		Returns:
			None
		"""
		if(simulation_time > self.__snp_creation_time >= 0):
			raise ValueError("Simulation time can't be larger than snp creation time.")
		
		if not self.exploded: # only update the simulation time if the star has not exploded yet
			time_since_snp_creation = self.snp_creation_time - simulation_time # how many years ago the snp was created
			if time_since_snp_creation >= self.expected_lifetime: # if True: the snp has exploded
				self.__age = self.expected_lifetime
				self.__exploded = True
				self.__sn_mass = 0 # the snp has exploded and has no mass left
				if self.__reference_to_companion is not None:
					if self.__expected_lifetime < self.__reference_to_companion.expected_lifetime: # if True: the companion star has the longest expected lifetime
						self._calculate_companion_final_position(self.__reference_to_companion, self.__expected_lifetime)
					else: # if False: the companion star has the longest expected lifetime
						self._calculate_companion_final_position(self, self.__reference_to_companion.expected_lifetime)
					# the star with the longest expected lifetime has recieved the kick. Remove both references to the companion star
					self.__reference_to_companion.__reference_to_companion = None 
					self.__reference_to_companion = None
			elif self.snp_creation_time < 0: # if True: the snp has not been created yet
				self.__age = self.snp_creation_time # the snp has not been created yet and naturally has no age
				self.__exploded = False
			else: # if False: the snp has not exploded 
				self.__age = time_since_snp_creation
				self.__exploded = False


	def _calculate_companion_final_position(self, existing_companion, age_exploded_companion):
		""" Calculate the point of death for the surviving companion star and update its position and velocity, based on the kick velocity and its initial velocity.
		
		Args:
			existing_companion (Supernovae): The remaining companion star.
			age_exploded_companion (int): The age of the exploded star at the time of explosion.
		
		Returns:
			None. Updates the position and velocity of the companion star, and sets its one_dim_vel = 0 to prevent _calculate_position() from updating the position again.
		"""
		# Update the position of the surviving companion star to the point where the other star exploded
		existing_companion.__age = age_exploded_companion
		existing_companion._calculate_position()
  
  		# Calculate the new speed and direction of the surviving companion star
		# Isotropic kick velocity direction
		theta_dir_kick = self.rng.uniform(0, np.pi)   # Random theta (0 to π)
		phi_dir_kick = self.rng.uniform(0, 2 * np.pi) # Random phi (0 to 2π)

		# Kick velocity magnitude (assumed from some distribution)
		kick_velocity = const.v_kick

		# Convert previous velocity to Cartesian components
		v_x_old = existing_companion.__one_dim_vel * np.sin(existing_companion.__vel_theta_dir) * np.cos(existing_companion.__vel_phi_dir)
		v_y_old = existing_companion.__one_dim_vel * np.sin(existing_companion.__vel_theta_dir) * np.sin(existing_companion.__vel_phi_dir)
		v_z_old = existing_companion.__one_dim_vel * np.cos(existing_companion.__vel_theta_dir)

		# Convert kick velocity to Cartesian components
		v_x_kick = kick_velocity * np.sin(theta_dir_kick) * np.cos(phi_dir_kick)
		v_y_kick = kick_velocity * np.sin(theta_dir_kick) * np.sin(phi_dir_kick)
		v_z_kick = kick_velocity * np.cos(theta_dir_kick)

		# Compute new velocity components
		v_x_new = v_x_old + v_x_kick
		v_y_new = v_y_old + v_y_kick
		v_z_new = v_z_old + v_z_kick

		# Compute new velocity magnitude and direction
		v_new = np.sqrt(v_x_new**2 + v_y_new**2 + v_z_new**2)
		theta_new = np.arccos(v_z_new / v_new)  # Inverse cosine for θ
		phi_new = np.arctan2(v_y_new, v_x_new)  # atan2 ensures correct quadrant for φ

		# Store the new velocity and direction
		existing_companion.__one_dim_vel = v_new
		existing_companion.__vel_theta_dir = theta_new
		existing_companion.__vel_phi_dir = phi_new

		# Update the position of the surviving companion star
		remaining_lifetime = existing_companion.expected_lifetime - age_exploded_companion
		existing_companion.__age = remaining_lifetime
		existing_companion._calculate_position()

		# Set velocity to zero so `_calculate_position()` does nothing
		existing_companion.one_dim_vel = 0


	def _calculate_position(self):
		""" Function to calculate the position of the supernova. The position is calculated in Cartesian coordinates (x, y, z) (kpc) and the longitude (radians).
		Updates the private attributes __sn_x, __sn_y, __sn_z and __long."""
		if self.__age < 0 or self.__one_dim_vel == 0:
			# Skip updating if the star hasn't formed yet or if it's already at its final position
			# A negative age means the snp has not been created yet, but is scheduled to be created in the future. Do not update the position
			return
		r = self.__one_dim_vel * const.seconds_in_myr * const.km_in_kpc * self.age # radial distance travelled by the supernova in kpc
		self.__sn_x += r * np.sin(self.vel_theta_dir) * np.cos(self.vel_phi_dir) # kpc
		self.__sn_y += r * np.sin(self.vel_theta_dir) * np.sin(self.vel_phi_dir) # kpc
		self.__sn_z += r * np.cos(self.vel_theta_dir) # kpc
		self.__long = (np.arctan2(self.y - const.r_s, self.x) + np.pi/2) % (2 * np.pi) # radians


	def update_snp(self, simulation_time):
		""" Method to update the snp. The age and position of the snp are updated for the given value for simulation_time.
		
		Args:
			simulation_time (int): The current time of the simulation in units of Myr. In the simulation, this will decrease on each iteration, counting down from the creation_time to the present
		
		Returns:
			None
		"""
		self._calculate_age(simulation_time)
		self._calculate_position()

	def set_companion(self, companion):
		""" Method to set the reference to the companion star in the binary system.
		
		Args:
			companion (Supernovae): The reference to the companion star in the binary system
		
		Returns:
			None
		"""
		self.__reference_to_companion = companion

	def plot_sn(self, ax):
		""" Function to plot the SNP on an ax object, relative to the association centre. Positions are converted to pc for the plot.
		
		Args:
			ax (matplotlib.axes.Axes): The ax object on which to plot the supernova.
		
		Returns:
			None
		"""
		if self.age < 0:
			# a negative age means the snp has not been created yet, but is scheduled to be created in the future. Do not plot it
			pass
		if self.exploded:
			ax.scatter((self.x - self.__association_x) * 1e3, (self.y - self.__association_y) * 1e3, (self.z - self.__association_z) * 1e3, c='r', s=5)
		else:
			ax.scatter((self.x - self.__association_x) * 1e3, (self.y - self.__association_y) * 1e3, (self.z - self.__association_z) * 1e3, c='black', s=1)
	
	
	def print_sn(self):
		if self.age < 0:
			# a negative age means the snp has not been created yet, but is scheduled to be created in the future. Do not print it
			pass
		print(f"SNP is located at xyz position ({self.x}, {self.y}, {self.z}). Mass: {self.mass}, lifetime: {self.age} yrs, bool_exploded: {self.exploded}.")
