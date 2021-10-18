import numpy as np


EPSILON = 1e-10
MAX_FLOAT = float('inf')

class AntColonyOptimizer:
	"""Ant Colony Optimizer"""
	def __init__(
		self, 
		radius,		
		pheromone_weight=1.0, 
		volatilization=0.5,
		heuristic_weight=3.0, 
		pheromone_per_ant=100, 
		*args, 
        **kwargs):
		# super(AntColonyOptimizer, self).__init__()
		
		self.radius = radius
		self.pheromone_weight = pheromone_weight
		self.volatilization = volatilization
		self.heuristic_weight = heuristic_weight
		self.pheromone_per_ant = pheromone_per_ant

	def _initialize_meshgrid(self, num_ant_per_dim, independent_variable_range):
		num_independent_variable = len(independent_variable_range)
		grids = [np.linspace(v_range[0], v_range[1], num_ant_per_dim) for v_range in independent_variable_range]		
		return np.array(np.meshgrid(*grids)).reshape(num_independent_variable, num_ant_per_dim ** num_independent_variable)

	def _min_max_normalize(self, array):

		min_element = np.min(array)
		max_element = np.max(array)
		return (array - min_element) / (max_element - min_element) + EPSILON

	def optimize(
		self, 
		num_independent_variable,
		independent_variable_range,
		objective_function,
		initial_pheromone=2.0, 
		initial_ant_position=None,
		max_iteration=200,
		num_ant_per_dim=5,
		tolerance=10, 
		local_search_threshold=0.02):

		num_ant = num_ant_per_dim ** num_independent_variable

		if initial_ant_position is None:
			assert len(independent_variable_range) == num_independent_variable, 'Please provide valid domain for each independent variable. Enter [] for domain R. '
			for i, v_range in enumerate(independent_variable_range):
				if not v_range:
					independent_variable_range[i] = [- MAX_FLOAT, MAX_FLOAT]
			ant_positions = self._initialize_meshgrid(num_ant_per_dim, independent_variable_range)
			# dimension x num_ant
		else:
			ant_positions = np.array(initial_ant_position)
				
		ant_indices = np.arange(num_ant)
		transition = np.zeros((num_ant, num_ant))
		neighbor_pheromone = initial_pheromone * np.ones(num_ant)		
		obj_values = objective_function(*ant_positions).reshape(1, -1)
		
		mean_obj_values = []
		min_obj_values = []
		objective_function_diff = obj_values - obj_values.T
			
		iteration = 1
		min_obj_value = np.min(obj_values)
		min_independent_variable = ant_positions[:, np.argmin(obj_values)]
		last_min_obj_value = MAX_FLOAT

		convergence = 0

		while iteration <= max_iteration:
		
			last_min_obj_value = min_obj_value
			delta_neighbor_pheromone = np.zeros(neighbor_pheromone.shape)

			for ant_index in ant_indices:											
				transition[ant_index] = (neighbor_pheromone ** self.pheromone_weight) * (objective_function_diff[ant_index] ** self.heuristic_weight)
				transition[ant_index] = self._min_max_normalize(transition[ant_index])
				transition[ant_index] /= np.sum(transition[ant_index])
				
				print(transition[ant_index])
				next_ant_neighbor_index = np.random.choice(ant_indices, p=transition[ant_index])
				current_neighbor_position = ant_positions[:, next_ant_neighbor_index]
				
				if objective_function_diff[ant_index, next_ant_neighbor_index] < 0:
					
					if transition[ant_index, next_ant_neighbor_index] < local_search_threshold:
						print('random walk in domain')
						sample_max = []
						sample_min = []
						for i, v_range in enumerate(independent_variable_range):
							sample_min.append(v_range[0])
							sample_max.append(v_range[1])
						current_neighbor_position = np.random.uniform(sample_min, sample_max)						

					if transition[ant_index, next_ant_neighbor_index] >= local_search_threshold or next_ant_neighbor_index == ant_index:
						print('search in current neighbor')						
						# search in current neighbor
						current_neighbor_position = ant_positions[:, ant_index]						
						search_radius = self.radius
						sample_max = []
						sample_min = []
						for i, v_range in enumerate(independent_variable_range):
							sample_min.append(max(v_range[0], current_neighbor_position[i] - search_radius))
							sample_max.append(min(v_range[1], current_neighbor_position[i] + search_radius))												
						current_neighbor_position = np.random.uniform(sample_min, sample_max)																							
				else:
					print('min value drops')
				ant_positions[:, ant_index] = current_neighbor_position
				current_obj_value = objective_function(*current_neighbor_position)
				
				if min_obj_value > current_obj_value:
					min_obj_value = current_obj_value
					min_independent_variable = current_neighbor_position

				delta_neighbor_pheromone[next_ant_neighbor_index] += self.pheromone_per_ant / (obj_values[0, ant_index] - current_obj_value + EPSILON)
				obj_values[0, ant_index] = current_obj_value

			mean_obj_values.append(np.mean(obj_values))
			min_obj_values.append(min_obj_value)
			objective_function_diff = obj_values - obj_values.T

			neighbor_pheromone = (1 - self.volatilization) * neighbor_pheromone + delta_neighbor_pheromone

			if np.abs(last_min_obj_value - min_obj_value) <= EPSILON:
				convergence += 1

			print('convergence %d, iteration %d, opt value %.6f' % (convergence, iteration, min_obj_value))

			iteration += 1


		return min_obj_value, min_independent_variable, mean_obj_values, min_obj_values



				
if __name__ == '__main__':

	opt = AntColonyOptimizer(radius=0.2)

	def obj_fun(x, y):

		return 20 * (x ** 2 - y ** 2) ** 2 - (1 - y) ** 2 - 3 * (1 + y) ** 2 + 0.3

	min_obj_value, min_independent_variable, _, _ = opt.optimize(
		num_ant_per_dim=3,
		num_independent_variable=2, 
		independent_variable_range=[[-3, 3], [-5, 5]], 
		objective_function=obj_fun)

	print(min_obj_value)
	print(min_independent_variable)


				










		
		