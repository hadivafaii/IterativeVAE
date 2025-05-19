from utils.plotting import *


class _Activation(nn.Module):
	def __init__(
			self,
			n_latents: int,
			fit_params: bool = True,
	):
		super(_Activation, self).__init__()
		self.n_latents = n_latents
		self.fit_params = fit_params
		self.log_gain = nn.Parameter(  # output gain
			data=torch.zeros(1, n_latents),
			requires_grad=self.fit_params,
		)

	def forward(self, z):
		raise NotImplementedError

	def derivative(self, z):
		raise NotImplementedError

	def show(self, idx=0, x_range=None, n_points=1000, figsize=(10, 4)):
		"""
		Plot the activation function and its derivative.

		Parameters:
		-----------
		x_range : tuple or None
			Range of x values to plot (min_x, max_x). If None, automatically determines a reasonable range.
		n_points : int
			Number of points to evaluate for the plot.
		latent_idx : int
			Index of the latent dimension to plot (for activations with multiple dimensions).
		figsize : tuple
			Figure size as (width, height) in inches.

		Returns:
		--------
		fig : matplotlib.figure.Figure
			The created figure.
		"""

		# If no range is provided, determine a reasonable default
		if x_range is None:
			# For most activation functions, [-5, 5] is a reasonable range
			x_range = (-5, 5)

			# For specific activation classes, customize the range
			class_name = self.__class__.__name__
			if class_name == 'ReLu':
				x_range = (-2, 5)
			elif class_name == 'Softplus':
				x_range = (-3, 5)
			elif class_name == 'FitzHughNagumo':
				x_range = (-2, 2)  # Cubic function is interesting in smaller range

		# Create evenly spaced points for evaluation
		x = np.linspace(x_range[0], x_range[1], n_points)
		x_tensor = torch.tensor(x).reshape(n_points, 1).to(
			device=self.log_gain.device,
			dtype=torch.float32,
		)
		# Evaluate the activation function and its derivative
		with torch.no_grad():
			y_forward = tonp(self.forward(x_tensor)[:, idx])
			y_derivative = tonp(self.derivative(x_tensor)[:, idx])

		# Create the plot
		fig, axes = plt.subplots(1, 2, figsize=figsize)

		# Plot the activation function
		axes[0].plot(x, y_forward)
		axes[0].set_title(f"{self.__class__.__name__} Activation")
		axes[0].set_xlabel("Input")
		axes[0].set_ylabel("Output")
		axes[0].grid(True)

		# Plot the derivative
		axes[1].plot(x, y_derivative)
		axes[1].set_title(f"{self.__class__.__name__} Derivative")
		axes[1].set_xlabel("Input")
		axes[1].set_ylabel("Derivative")
		axes[1].grid(True)

		# Add parameter information to the figure
		param_text = "Parameters:\n"
		for name, param in self.named_parameters():
			if idx < param.shape[1]:
				# For vector parameters, show the value at latent_idx
				value = param.data[0, idx].item()
				if 'log_' in name:
					# For log parameters, also show the exponentiated value
					exp_value = np.exp(value)
					param_text += f"{name} = {value:.4f} (exp: {exp_value:.4f})\n"
				else:
					param_text += f"{name} = {value:.4f}\n"

		# Add the parameter text at the bottom of the figure
		fig.text(
			0.5, 0.01, param_text,
			horizontalalignment='center',
			verticalalignment='bottom',
			bbox=dict(facecolor='white', alpha=0.8),
		)

		plt.tight_layout()
		plt.subplots_adjust(bottom=0.2)  # Make room for the parameter text
		return fig


class ReLu(_Activation):
	def __init__(self, n_latents: int, **kwargs):
		super(ReLu, self).__init__(
			n_latents, **kwargs)

	def forward(self, z):
		z = F.relu(z)
		return z * self.log_gain.exp()

	def derivative(self, z):
		return z.gt(0).to(z.dtype) * self.log_gain.exp()


class Logistic(_Activation):
	def __init__(
			self,
			n_latents: int,
			alpha_init: float = 1.0,
			beta_init: float = 0.0,
			**kwargs,
	):
		super(Logistic, self).__init__(
			n_latents, **kwargs)

		# alpha: steepness parameter
		self.log_alpha = nn.Parameter(
			data=torch.ones(1, n_latents) * np.log(alpha_init),
			requires_grad=self.fit_params,
		)
		# beta: shift parameter
		self.beta = nn.Parameter(
			data=torch.ones(1, n_latents) * beta_init,
			requires_grad=self.fit_params,
		)

	def forward(self, z):
		# logistic function: gain * 1 / (1 + exp(-α(x-β)))
		g = torch.exp(self.log_gain)
		a = torch.exp(self.log_alpha)
		return g / (1.0 + torch.exp(-a * (z - self.beta)))

	def derivative(self, z):
		# derivative: gain * α * f(x) * (1 - f(x))
		g = torch.exp(self.log_gain)
		a = torch.exp(self.log_alpha)
		fx_base = 1.0 / (1.0 + torch.exp(-a * (z - self.beta)))
		return g * a * fx_base * (1.0 - fx_base)


class Gompertz(_Activation):
	def __init__(
			self,
			n_latents: int,
			b_init: float = 1.0,  # displacement parameter
			c_init: float = 1.0,  # growth rate parameter
			**kwargs,
	):
		super(Gompertz, self).__init__(
			n_latents, **kwargs)
		# b: displacement parameter (shifts curve along x-axis)
		self.log_b = nn.Parameter(
			data=torch.ones(1, n_latents) * np.log(b_init),
			requires_grad=self.fit_params,
		)
		# c: growth rate parameter (controls slope)
		self.log_c = nn.Parameter(
			data=torch.ones(1, n_latents) * np.log(c_init),
			requires_grad=self.fit_params,
		)

	def forward(self, z):
		# Gompertz function: gain * exp(-b * exp(-c * x))
		g = torch.exp(self.log_gain)
		b = torch.exp(self.log_b)
		c = torch.exp(self.log_c)
		return g * torch.exp(-b * torch.exp(-c * z))

	def derivative(self, z):
		# derivative: gain * b * c * exp(-c * x - b * exp(-c * x))
		g = torch.exp(self.log_gain)
		b = torch.exp(self.log_b)
		c = torch.exp(self.log_c)
		exp_neg_cz = torch.exp(-c * z)
		exp_neg_b_exp = torch.exp(-b * exp_neg_cz)
		return g * b * c * exp_neg_cz * exp_neg_b_exp


class Softplus(_Activation):
	def __init__(
			self,
			n_latents: int,
			beta_init: float = 1.0,  # smoothing parameter
			**kwargs,
	):
		super(Softplus, self).__init__(
			n_latents, **kwargs)
		# beta: controls the smoothness of the curve
		self.log_beta = nn.Parameter(
			data=torch.ones(1, n_latents) * np.log(beta_init),
			requires_grad=self.fit_params,
		)

	def forward(self, z):
		# Softplus function: gain * ln(1 + exp(beta * x)) / beta
		g = torch.exp(self.log_gain)
		beta = torch.exp(self.log_beta)

		# Manual implementation to allow for per-element beta
		# softplus(x) = ln(1 + exp(beta * x)) / beta
		z_scaled = beta * z

		# For numerical stability, use different computation for large values
		result = torch.zeros_like(z)
		mask = z_scaled > 20  # threshold for numerical stability

		# For large inputs, softplus(x) ≈ x
		result[mask] = z[mask]

		# For smaller inputs, use standard formula
		safe_exp = torch.exp(z_scaled[~mask])
		result[~mask] = torch.log(1 + safe_exp) / beta[~mask]

		return g * result

	def derivative(self, z):
		# Derivative of softplus: gain * sigmoid(beta * x)
		g = torch.exp(self.log_gain)
		beta = torch.exp(self.log_beta)

		return g * torch.sigmoid(beta * z)


class GaussianCDF(_Activation):
	def __init__(
			self,
			n_latents: int,
			mu_init: float = 0.0,
			sigma_init: float = 1.0,
			**kwargs,
	):
		super(GaussianCDF, self).__init__(
			n_latents, **kwargs)
		# mu: mean parameter (shift)
		self.mu = nn.Parameter(
			data=torch.ones(1, n_latents) * mu_init,
			requires_grad=self.fit_params,
		)
		# sigma: standard deviation parameter (width)
		self.log_sigma = nn.Parameter(
			data=torch.ones(1, n_latents) * np.log(sigma_init),
			requires_grad=self.fit_params,
		)

	def _normalized_input(self, z):
		sigma = torch.exp(self.log_sigma)
		return (z - self.mu) / sigma

	def forward(self, z):
		# Gaussian CDF: gain * 0.5 * (1 + erf((x - mu)/(sigma * sqrt(2))))
		g = torch.exp(self.log_gain)
		z_norm = self._normalized_input(z)
		return g * 0.5 * (1 + torch.erf(z_norm / np.sqrt(2)))

	def derivative(self, z):
		# Derivative: gain * exp(-((x-mu)/sigma)^2/2) / (sigma * sqrt(2π))
		g = torch.exp(self.log_gain)
		sigma = torch.exp(self.log_sigma)
		z_norm = self._normalized_input(z)
		return g * torch.exp(-z_norm ** 2 / 2) / (sigma * np.sqrt(2 * np.pi))


class ArctanSigmoid(_Activation):
	def __init__(
			self,
			n_latents: int,
			alpha_init: float = 1.0,  # steepness parameter
			beta_init: float = 0.0,   # shift parameter
			**kwargs,
	):
		super(ArctanSigmoid, self).__init__(
			n_latents, **kwargs)
		# alpha: steepness parameter
		self.log_alpha = nn.Parameter(
			data=torch.ones(1, n_latents) * np.log(alpha_init),
			requires_grad=self.fit_params,
		)
		# beta: shift parameter
		self.beta = nn.Parameter(
			data=torch.ones(1, n_latents) * beta_init,
			requires_grad=self.fit_params,
		)

	def forward(self, z):
		# Arctan sigmoid: gain * ((1/pi) * arctan(alpha * (x - beta)) + 0.5)
		g = torch.exp(self.log_gain)
		a = torch.exp(self.log_alpha)

		return g * ((1 / np.pi) * torch.atan(a * (z - self.beta)) + 0.5)

	def derivative(self, z):
		# Derivative: gain * alpha / (pi * (1 + (alpha * (x - beta))^2))
		g = torch.exp(self.log_gain)
		a = torch.exp(self.log_alpha)
		scaled_input = a * (z - self.beta)
		return g * a / (np.pi * (1 + scaled_input ** 2))


class Tanh(_Activation):
	def __init__(
			self,
			n_latents: int,
			alpha_init: float = 1.0,  # steepness parameter
			beta_init: float = 0.0,  # shift parameter
			**kwargs,
	):
		super(Tanh, self).__init__(
			n_latents, **kwargs)
		# alpha: steepness parameter
		self.log_alpha = nn.Parameter(
			data=torch.ones(1, n_latents) * np.log(alpha_init),
			requires_grad=self.fit_params,
		)
		# beta: shift parameter
		self.beta = nn.Parameter(
			data=torch.ones(1, n_latents) * beta_init,
			requires_grad=self.fit_params,
		)

	def forward(self, z):
		# Tanh function: gain * tanh(alpha * (x - beta))
		g = torch.exp(self.log_gain)
		a = torch.exp(self.log_alpha)
		return g * torch.tanh(a * (z - self.beta))

	def derivative(self, z):
		# Derivative: gain * alpha * (1 - tanh^2(alpha * (x - beta)))
		g = torch.exp(self.log_gain)
		a = torch.exp(self.log_alpha)
		tanh_val = torch.tanh(a * (z - self.beta))
		return g * a * (1 - tanh_val ** 2)


class NakaRushton(_Activation):
	def __init__(
			self,
			n_latents: int,
			n_init: float = 2.0,
			c50_init: float = 1.0,
			**kwargs,
	):
		super(NakaRushton, self).__init__(
			n_latents, **kwargs)
		# n: exponent parameter
		self.log_n = nn.Parameter(
			data=torch.ones(1, n_latents) * np.log(n_init),
			requires_grad=self.fit_params,
		)
		# c50: semi-saturation constant
		self.log_c50 = nn.Parameter(
			data=torch.ones(1, n_latents) * np.log(c50_init),
			requires_grad=self.fit_params,
		)

	def forward(self, z):
		# Ensure inputs are positive
		z_pos = torch.abs(z)
		g = torch.exp(self.log_gain)
		c50 = torch.exp(self.log_c50)
		n = torch.exp(self.log_n)
		return g * (z_pos ** n) / (z_pos ** n + c50 ** n)

	def derivative(self, z):
		g = torch.exp(self.log_gain)
		c50 = torch.exp(self.log_c50)
		n = torch.exp(self.log_n)
		z_pos = torch.abs(z)
		# Use chain rule to compute derivative
		numerator = n * c50 ** n * z_pos ** (n - 1)
		denominator = (z_pos ** n + c50 ** n) ** 2
		return g * numerator / denominator * torch.sign(z)


class FitzHughNagumo(_Activation):
	def __init__(
			self,
			n_latents: int,
			a_init: float = 0.8,
			b_init: float = 0.7,
			**kwargs,
	):
		super(FitzHughNagumo, self).__init__(
			n_latents, **kwargs)
		# a: threshold parameter
		self.a = nn.Parameter(
			data=torch.ones(1, n_latents) * a_init,
			requires_grad=self.fit_params,
		)
		# b: recovery parameter (0 < b < 1)
		self.log_b = nn.Parameter(
			data=torch.ones(1, n_latents) * np.log(b_init),
			requires_grad=self.fit_params,
		)

	def forward(self, z):
		g = torch.exp(self.log_gain)
		b = torch.exp(self.log_b).clamp(0, 1)  # ensure 0 < b < 1
		# Simplified FitzHugh-Nagumo-like activation
		z_cube = z ** 3 / 3
		return g * (z - z_cube - self.a) / (1 + b)

	def derivative(self, z):
		g = torch.exp(self.log_gain)
		b = torch.exp(self.log_b).clamp(0, 1)
		return g * (1 - z ** 2) / (1 + b)


class AdaptiveExponential(_Activation):
	def __init__(
			self,
			n_latents: int,
			v_thresh_init: float = 1.0,
			delta_t_init: float = 0.2,
			**kwargs,
	):
		super(AdaptiveExponential, self).__init__(
			n_latents, **kwargs)
		# v_thresh: threshold
		self.v_thresh = nn.Parameter(
			data=torch.ones(1, n_latents) * v_thresh_init,
			requires_grad=self.fit_params,
		)
		# delta_t: sharpness
		self.log_delta_t = nn.Parameter(
			data=torch.ones(1, n_latents) * np.log(delta_t_init),
			requires_grad=self.fit_params,
		)

	def forward(self, z):
		g = torch.exp(self.log_gain)
		delta_t = torch.exp(self.log_delta_t)
		# Compute exponential term with safeguards
		exp_term = torch.exp((z - self.v_thresh) / delta_t)
		# Apply sigmoid to bound the result
		return g * torch.sigmoid(z + delta_t * exp_term - self.v_thresh)

	def derivative(self, z):
		g = torch.exp(self.log_gain)
		delta_t = torch.exp(self.log_delta_t)
		exp_term = torch.exp((z - self.v_thresh) / delta_t)
		sigmoid_input = z + delta_t * exp_term - self.v_thresh
		sigmoid_output = torch.sigmoid(sigmoid_input)
		return g * sigmoid_output * (1 - sigmoid_output) * (1 + exp_term)


ACT_CLASSES = {
	'relu': ReLu,
	'logistic': Logistic,
	'gompertz': Gompertz,
	'softplus': Softplus,
	'gcdf': GaussianCDF,
	'arctan': ArctanSigmoid,
	'tanh': Tanh,
	'nakarush': NakaRushton,
	'fitznagu': FitzHughNagumo,
	'adaexp': AdaptiveExponential,
	# 'ricciardi': Ricciardi,
}


#######################
# Activation functions
#######################

def smooth_relu_derivative(temp: float = 1e-2):
	def _fun(x):
		return torch.sigmoid(x / temp)
	return _fun


def sigmoid_derivative(x):
	s = torch.sigmoid(x)
	return s * (1 - s)


def tanh_derivative(x):
	x = torch.tanh(x)
	return 1 - x ** 2


def gaussian_cdf(x):
	return 0.5 * (1 + torch.erf(x / np.sqrt(2)))


def gaussian_cdf_derivative(x):
	return torch.exp(-x**2 / 2) / np.sqrt(2 * np.pi)


def gompertz(x):
	return torch.exp(-torch.exp(-x))


def gompertz_derivative(x):
	return torch.exp(-x - torch.exp(-x))


def arctan_sigmoid(x):
	# f(x) = (1/pi)*arctan(x) + 1/2
	return (1 / np.pi) * torch.atan(x) + 0.5


def arctan_sigmoid_derivative(x):
	return 1 / (np.pi * (1 + x**2))


# activations: (function, derivative)
ACT_FUNS = {
	'relu': (
		F.relu,
		smooth_relu_derivative()),
	'exp': (
		torch.exp,
		torch.exp),
	'softplus': (
		F.softplus,
		torch.sigmoid),
	'sigmoid': (
		torch.sigmoid,
		sigmoid_derivative),
	'gcdf': (
		gaussian_cdf,
		gaussian_cdf_derivative),
	'tanh': (
		torch.tanh,
		tanh_derivative),
	'arctan': (
		arctan_sigmoid,
		arctan_sigmoid_derivative),
	'gompertz': (
		gompertz,
		gompertz_derivative),
	'square': (
		torch.square,
		lambda x: 2 * x),
	'quartic': (
		lambda x: x.pow(4),
		lambda x: 4 * x.pow(3)),
}
