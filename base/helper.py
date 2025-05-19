from utils.generic import *


def rotate_img(
		x: torch.Tensor,
		angles: float | Iterable[float],
		interpolation: str = 'bilinear',
		hist_match: bool = True, ):
	if x.ndim == 4:
		batched = True
	elif x.ndim == 3:
		batched = False
	else:
		msg = f"Unsupported dims: {x.ndim}"
		raise ValueError(msg)

	if not isinstance(angles, Iterable):
		angles = [angles]
		if batched:
			angles = angles * len(x)

	if batched:
		assert len(angles) == len(x)

	if all(theta % 360 == 0 for theta in angles):
		return x

	interp_mode = getattr(
		F_vis.InterpolationMode,
		interpolation.upper(),
		F_vis.InterpolationMode.BILINEAR,
	)

	if batched:
		if all(theta == angles[0] for theta in angles):
			x_rotated = F_vis.rotate(
				inpt=x,
				angle=angles[0],
				interpolation=interp_mode,
			)
		else:
			rotated_list = [
				F_vis.rotate(
					inpt=img,
					angle=theta,
					interpolation=interp_mode,
				) for theta, img in zip(angles, x)
			]
			x_rotated = torch.stack(
				rotated_list, dim=0)
	else:
		# otherwise, x is assumed to
		# be a single image (C, H, W)
		x_rotated = F_vis.rotate(
			inpt=x,
			angle=angles[0],
			interpolation=interp_mode,
		)

	# optionally: apply histogram matching
	if hist_match:
		x_rotated = match_histograms(
			src=x_rotated, ref=x)

	return x_rotated


def match_histograms(
		src: torch.Tensor,
		ref: torch.Tensor, ) -> torch.Tensor:
	orig_dims = src.ndim
	# ensure that both src and ref have
	# four dims (B, C, H, W) for processing
	while src.ndim < 4:
		src = src.unsqueeze(0)
		ref = ref.unsqueeze(0)
	result = match_histograms_batch(
		src=src, ref=ref)
	while result.ndim > orig_dims:
		result = result.squeeze(0)
	return result


def match_histograms_batch(
		src: torch.Tensor,
		ref: torch.Tensor, ) -> torch.Tensor:
	b, c, h, w = src.shape
	s = src.reshape(b, c, -1)
	r = ref.reshape(b, c, -1)
	s_sorted, s_indices = torch.sort(s, dim=2)
	r_sorted, _ = torch.sort(r, dim=2)
	s_matched = torch.empty_like(s)
	s_matched.scatter_(2, s_indices, r_sorted)
	return s_matched.reshape(b, c, h, w)


def translate_img(
		x: torch.Tensor,
		translate_x: float,
		translate_y: float,
		interpolation: str = 'bilinear',
		hist_match: bool = True, ):
	if translate_x == translate_y == 0:
		return x
	interpolation = getattr(
		F_vis.InterpolationMode,
		interpolation.upper(),
	)
	x_translated = F_vis.affine(
		inpt=x,
		angle=0.0,
		scale=1.0,
		shear=[0.0],
		translate=[translate_x, translate_y],
		interpolation=interpolation,
	)
	# optionally: apply histogram matching
	if hist_match:
		x_translated = match_histograms(
			src=x_translated, ref=x)
	return x_translated


def job_runner_script(
		device: int,
		dataset: str,
		seed: int = 0,
		args: str = None,
		model: str = 'poisson',
		bash_script: str = 'fit_model.sh',
		relative_path: str = '.', ):
	s = ' '.join([
		f"'{device}'",
		f"'{dataset}'",
		f"'{model}'",
	])
	s = f"{relative_path}/{bash_script} {s}"
	s = f"{s} --seed {seed}"
	if args is not None:
		s = f"{s} {args}"
	return s


def skew(x: np.ndarray, axis: int = 0):
	x1 = np.expand_dims(np.expand_dims(np.take(
		x, 0, axis=axis), axis=axis), axis=axis)
	x2 = np.expand_dims(np.expand_dims(np.take(
		x, 1, axis=axis), axis=axis), axis=axis)
	x3 = np.expand_dims(np.expand_dims(np.take(
		x, 2, axis=axis), axis=axis), axis=axis)
	s1 = np.concatenate([np.zeros_like(x1), -x3, x2], axis=axis+1)
	s2 = np.concatenate([x3, np.zeros_like(x2), -x1], axis=axis+1)
	s3 = np.concatenate([-x2, x1, np.zeros_like(x3)], axis=axis+1)
	s = np.concatenate([s1, s2, s3], axis=axis)
	return s


# noinspection PyTypeChecker
def find_convergence(
		x: np.ndarray,
		window: int = 20,
		tol: float = 1e-5,
		consecutive: int = 5, ):
	x = np.asarray(x)
	slopes = []
	t_axis = np.arange(window)
	denom = np.var(t_axis) * window
	for i in range(len(x) - window + 1):
		y = x[i:i + window]
		cov = np.cov(t_axis, y, bias=True)[0, 1]
		slopes.append(cov / denom)
	slopes = np.abs(np.array(slopes))

	below = slopes < tol
	runlen = np.convolve(below, np.ones(consecutive, dtype=int), mode='valid')
	idx = np.where(runlen == consecutive)[0]
	if idx.size:
		return int(idx[0] + window)
	return len(x) - 1


def find_weight_concentrations(
		w: np.ndarray | torch.Tensor,
		distance_fn: Callable = None, ):
	"""
	Identify the center of mass for each weight matrix and order them
	based on a custom distance function (defaults to distance from origin).

	Args:
		w: numpy array of shape (n_weights, height, width)
		distance_fn: optional function that takes a tuple (i, j)
			and returns a distance metric (defaults to Euclidean
			distance from origin)

	Returns:
		ordered_indices: list of indices ordering the weights
		centers: list of (i, j) coordinates representing center
		of mass for each weight
	"""
	n_weights, height, width = w.shape
	centers = []

	# Default distance function: Euclidean distance from origin (0,0)
	if distance_fn is None:
		distance_fn = lambda point: np.sqrt(point[0] ** 2 + point[1] ** 2)

	# Calculate center of mass for each weight
	for w_idx in range(n_weights):
		w_abs = np.abs(w[w_idx])

		i_coords, j_coords = np.meshgrid(
			np.arange(height),
			np.arange(width),
			indexing='ij',
		)

		# Calculate total weight
		total_weight = np.sum(w_abs)

		# Default to center if weight is all zeros
		if total_weight == 0:
			centers.append((height // 2, width // 2))
			continue

		# Calculate weighted average of coordinates
		center_i = np.sum(i_coords * w_abs) / total_weight
		center_j = np.sum(j_coords * w_abs) / total_weight

		centers.append((center_i, center_j))

	# Calculate distances using the provided distance function
	distances = [
		(i, distance_fn(center)) for
		i, center in enumerate(centers)
	]

	# Sort weights by distance
	distances.sort(key=lambda x: x[1])
	ordered_indices = [idx for idx, _ in distances]

	ordered_indices = np.asarray(ordered_indices)
	centers = np.asarray(centers)
	return ordered_indices, centers


def conv_arithmetic(
		input_size: int = None,
		output_size: int = None,
		kernel_size: int = None,
		stride: int = 1,
		padding: int = 0,
		deconv: bool = False, ):
	"""
	Exactly one of the parameters must be None; the others must be provided.

	For conv (deconv=False), the relationship is:
		output_size = (input_size + 2 * padding - kernel_size) / stride + 1

	For deconv (deconv=True), the relationship is:
		output_size = (input_size - 1) * stride - 2 * padding + kernel_size
	"""

	# Put parameters in a list so we can count how many are None.
	params = [input_size, kernel_size, stride, padding, output_size]
	assert params.count(None) == 1, 'exactly one param must be None'

	# --- Conv ---
	if not deconv:
		# Formula: output_size = (input_size + 2*padding - kernel_size) / stride + 1
		if output_size is None:
			computed = (input_size + 2 * padding - kernel_size) / stride + 1
		elif input_size is None:
			computed = (output_size - 1) * stride - 2 * padding + kernel_size
		elif kernel_size is None:
			computed = input_size + 2 * padding - (output_size - 1) * stride
		elif stride is None:
			assert output_size > 1, 'otherwise we get division by zero'
			computed = (input_size + 2 * padding - kernel_size) / (output_size - 1)
		elif padding is None:
			computed = (((output_size - 1) * stride) - input_size + kernel_size) / 2
		else:
			raise RuntimeError(params.count(None))
	# --- Deconv ---
	else:
		# Formula: output_size = (input_size - 1)*stride - 2*padding + kernel_size
		if output_size is None:
			computed = (input_size - 1) * stride - 2 * padding + kernel_size
		elif input_size is None:
			assert stride > 0, 'otherwise we get division by zero'
			computed = ((output_size + 2 * padding - kernel_size) / stride) + 1
		elif kernel_size is None:
			computed = output_size - (input_size - 1) * stride + 2 * padding
		elif stride is None:
			assert input_size > 1, 'otherwise we get division by zero'
			computed = (output_size + 2 * padding - kernel_size) / (input_size - 1)
		elif padding is None:
			computed = (((input_size - 1) * stride) - output_size + kernel_size) / 2
		else:
			raise RuntimeError(params.count(None))

	assert int(computed) == computed
	return int(computed)


class ParameterGrid:
	"""
	A class to manage parameter combinations in a grid search.

	This class efficiently handles combinations of parameter values without
	explicitly storing all combinations, provides fast lookups from index to
	parameter values and vice versa.
	"""

	def __init__(self, parameter_dict: Dict[str, ...]):
		"""
		Initialize with a dictionary of parameter
		names and their possible values.

		Args:
			parameter_dict: Dictionary mapping parameter
				names to arrays of their values
		"""
		self.param_dict = parameter_dict
		self.param_values = [np.array(v) for v in parameter_dict.values()]
		self.param_lengths = [len(values) for values in self.param_values]

		# Calculate total number of combinations
		self.total_combinations = np.prod(self.param_lengths)

		# Calculate strides for each parameter (for efficient index conversion)
		self.strides = []
		stride = 1
		for length in reversed(self.param_lengths):
			self.strides.insert(0, stride)
			stride *= length

	def get_param_vectors(self):
		"""
		Generate vectors for each parameter, where each value
		is repeated according to the combinatorial structure.

		Returns:
			dict mapping parameter names to their expanded vectors
		"""
		param_vectors = {}

		for i, name in enumerate(self.param_dict):
			# Calculate repetition pattern
			# Each value repeats based on the
			# product of all lengths to its right
			repeats = np.prod(
				self.param_lengths[i + 1:]
			) if i < len(self.param_dict) - 1 else 1

			# Calculate tiles
			# The whole pattern repeats based on
			# the product of all lengths to its left
			tiles = np.prod(self.param_lengths[:i]) if i > 0 else 1

			# Create the vector
			param_vectors[name] = np.tile(np.repeat(
				self.param_values[i], repeats), tiles)

		return param_vectors

	def __len__(self):
		return self.total_combinations

	def __getitem__(self, idx: int | str):
		if isinstance(idx, int):
			return self._index_to_params(idx)
		elif isinstance(idx, str):
			return self.param_dict[idx]
		else:
			raise ValueError(type(idx))

	def _index_to_params(self, idx: int):
		"""
		Convert a flat index to the corresponding parameter values.

		Args:
			idx: Index in the range [0, total_combinations - 1]

		Returns:
			Dictionary mapping parameter names to their values
		"""
		if idx < 0 or idx >= self.total_combinations:
			max_idx = self.total_combinations - 1
			msg = f"Index must be between 0 and {max_idx}"
			raise ValueError(msg)

		params = {}
		for i, name in enumerate(self.param_dict):
			# Integer division gives the index into the parameter's values
			param_idx = (idx // self.strides[i]) % self.param_lengths[i]
			params[name] = self.param_values[i][param_idx]

		return params

	def get_param_at_index(self, idx, param_name):
		"""
		Get a specific parameter value at a given index.

		Args:
			idx: Index in the range [0, total_combinations - 1]
			param_name: Name of the parameter to retrieve

		Returns:
			The value of the specified parameter at the given index
		"""
		params = self._index_to_params(idx)
		return params[param_name]

	def params_to_index(self, **kwargs):
		"""
		Convert parameter values to the corresponding flat index.

		Args:
			**kwargs: Parameter values (e.g., contrast=0.2, s_freq=5)

		Returns:
			Index corresponding to these parameter values
		"""
		idx = 0
		for i, name in enumerate(self.param_dict):
			if name not in kwargs:
				raise ValueError(f"Missing parameter: {name}")

			# Find the closest parameter value
			value = kwargs[name]
			param_idx = np.abs(self.param_values[i] - value).argmin()

			# Add contribution to the index
			idx += param_idx * self.strides[i]

		return idx

	def get_all_combinations(self):
		"""
		Generate all parameter combinations (use with caution for large grids).

		Returns:
			List of dictionaries, each containing one parameter combination
		"""
		all_combinations = []
		for idx in range(self.total_combinations):
			all_combinations.append(self._index_to_params(idx))
		return all_combinations
