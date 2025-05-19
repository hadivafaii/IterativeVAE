from main.distributions import *
from .activations import ACT_CLASSES
from contextlib import contextmanager


class _StochasticLayer(nn.Module):
	def __init__(
			self,
			input_dim: int,
			latent_dim: int,
			cfg: ConfigVAE,
	):
		"""
		input_dim: input to be explained away
		latent_dim: also interpreted as output_dim
		"""
		super(_StochasticLayer, self).__init__()
		self.n_updates = 0
		self.time = 0
		self.u = None
		self.u_prior = None
		self.samples = None
		self.posterior = None
		self.conditional = None
		self.input_dim = input_dim
		self.latent_dim = latent_dim
		self._cfg = cfg
		self._init()

	def forward(self, x, temp: float = None):
		if self.time == 0 or self.u is None:
			self.reset_state(len(x))
		self.initialize_inference()
		if temp is None:
			return self.infer(x)
		with self.temporary_temperature(temp):
			return self.infer(x)

	def infer(self, x):
		if self._cfg.inf_type == 'grad':
			return self.infer_grad(x)
		elif self._cfg.inf_type == 'leapfrog':
			return self.infer_leapfrog(x)
		else:
			msg = 'invalid inference type: '
			msg = f"{msg}{self._cfg.inf_type}"
			raise NotImplementedError(msg)

	def infer_grad(self, x):
		for n in range(self._cfg.n_iters_inner):
			self.n_updates += 1
			grad_f = self.grad_free_energy(x)
			u_dot = tuple(
				llr.exp() * g.mul(-1.0) for g, llr
				in zip(grad_f, self.log_lr)
			)
			u_dot = tuple(
				self.softclamp(item, 'du', i)
				for i, item in enumerate(u_dot)
			)
			self.u = tuple(
				self.softclamp(u + du, 'u', i) for
				i, (u, du) in enumerate(zip(self.u, u_dot))
			)
		return self.posterior

	def generate(self, temp: float = None):
		self.update_inference_state(temp)
		z = self.apply_act_fn()
		self.conditional = Normal(
			loc=self.phi(z),
			log_scale=self.get_dec_log_sigma(),
			clamp=None,
			temp=1.0,
		)
		return self.conditional

	def pred_error(self, x, weighted: bool = True):
		pred_error = x - self.conditional.loc
		if weighted:
			var = self.conditional.scale ** 2
			pred_error = pred_error / var
		return pred_error

	def grad_recon_term(self, x):
		# (1) generate samples
		# updates: (self.posterior, self.samples)
		self.update_inference_state()
		# (2) apply activation fun
		z = self.apply_act_fn()
		# (3) decoder variance
		dec_log_sigma = self.get_dec_log_sigma()
		dec_var = torch.exp(2.0 * dec_log_sigma)
		# (4) grad: Recon. term
		error = (x - self.phi(z)) / dec_var
		phi_error = self.phi(error, transpose=True)
		# (4) apply activation derivative
		phi_error = self.apply_act_deriv_fn(phi_error)
		# (5) finalize
		grad_recon = phi_error.mul(-1.0)
		return grad_recon

	def grad_kl_term(self):
		raise NotImplementedError

	def grad_free_energy(self, x):
		raise NotImplementedError

	def estimate_acceleration(self, x):
		raise NotImplementedError

	def initialize_inference(self):
		if self.u is None:
			msg = "run reset_state() first"
			raise RuntimeError(msg)
		self.u_prior = tuple(u for u in self.u)
		return

	def reset_state(self, batch_size: int = None):
		size = (batch_size, -1)
		self.u = tuple(
			u0.expand(size=size)
			for u0 in self.u_init
		)
		self.n_updates = 0
		self.time = 0
		return

	def detach_state(self):
		self.u = tuple(u.detach() for u in self.u)
		return

	def attach_prior(self):
		size = (len(self.u[0]), -1)
		u_init = tuple(
			u0.expand(size=size)
			for u0 in self.u_init
		)
		self.u = tuple(
			u0 + (u - u0).detach() for
			u0, u in zip(u_init, self.u)
		)
		return

	def softclamp(self, x: torch.Tensor, which: str, idx: int):
		raise NotImplementedError

	def loss_kl(self, *args, **kwargs):
		raise NotImplementedError

	def loss_recon(self, x, mode: str = 'prob'):
		if mode == 'prob':
			loss = -self.conditional.log_prob(x)
			loss = torch.sum(loss, dim=-1)
		elif mode == 'mse':
			loss = (x - self.conditional.loc).pow(2)
			loss = torch.sum(loss, dim=-1)
		elif mode == 'r2':
			# don't use for training, only for eval
			loss = compute_r2(x, self.conditional.loc)
		else:
			raise NotImplementedError(mode)
		return loss

	def update_inference_state(self, temp: float = None):
		if temp is None:
			self.posterior = self.get_dist()
		else:
			with self.temporary_temperature(temp):
				self.posterior = self.get_dist()
		self.samples = self.posterior.rsample()
		return

	def apply_act_fn(self, z=None):
		if z is None:
			z = self.samples
		if self.activation is not None:
			return self.activation(z)
		return z

	def apply_act_deriv_fn(self, inpt, z=None):
		if z is None:
			z = self.samples
		if self.activation is not None:
			deriv = self.activation.derivative(z)
			return deriv * inpt
		return inpt

	def get_dec_log_sigma(self, c: float = 3.45):
		"""
		c = 1.5   —> var: (~0.05, ~20.0)
		c = 2.3   —> var: (~0.01, ~100.0)
		c = 3.45  —> var: (~0.001, ~1000.0)
		c = 4.6   —> var: (~0.0001, ~10,000.0)
		"""
		return softclamp_sym(self.dec_log_sigma, c)

	def get_c(self, which: str, idx: int = None):
		assert which in ['u', 'du']
		c_final = getattr(self._cfg, f"clamp_{which}")
		if isinstance(c_final, Sequence) and idx is not None:
			c_final = c_final[idx]
		c_init = 2.0 if self._cfg.type == 'poisson' else 1.0
		c = c_init + self.eps * (c_final - c_init)
		return c

	def get_dist(self, posterior: bool = True):
		raise NotImplementedError

	def get_weight(self):
		w = self.phi.weight.data.clone()
		return w

	@contextmanager
	def temporary_temperature(self, temp):
		temp_orig = self.temp.item()     # store the original
		self.update_temp(temp)           # set new temperature
		try:
			yield  # run whatever needs the new temp
		finally:
			self.update_temp(temp_orig)  # revert back

	def update_temp(self, new_t: float):
		if new_t is None:
			return
		assert new_t >= 0.0, "must be non-neg"
		self.temp.fill_(new_t)
		return

	def update_beta(self, new_beta: float):
		if new_beta is None:
			return
		assert new_beta >= 0.0, "must be non-neg"
		self.beta_inner.fill_(new_beta)
		return

	def update_eps(self, new_eps: float):
		if new_eps is None:
			return
		assert new_eps >= 0.0, "must be non-neg"
		self.eps.fill_(new_eps)
		return

	def _init(self):
		# self._init_enc()
		self.t_total = (
			self._cfg.t_train *
			self._cfg.n_iters_outer
		)
		self._init_dec()
		self._init_act()
		self._init_inner()
		self._init_weight()
		self._init_buffers()
		self._init_prior()
		return

	def _init_enc(self):
		raise NotImplementedError

	def _init_dec(self):
		# phi shape: [M x K]
		self.phi = LinearDictionary(
			dim_1=self.input_dim,
			dim_2=self.latent_dim,
		)
		# decoder: log scale param
		dec_log_sigma = torch.zeros(
			(1, self.input_dim),
			dtype=torch.float,
		)
		self.dec_log_sigma = nn.Parameter(
			data=dec_log_sigma,
			requires_grad=True,
		)
		return

	def _init_act(self):
		self.activation = None
		act_name = getattr(self._cfg, 'latent_act', None)
		act_class = ACT_CLASSES.get(act_name, None)
		if act_class is not None:
			self.activation = act_class(
				n_latents=self.latent_dim,
				fit_params=True,
			)
		return

	def _init_inner(self):
		"""
		All used in the inner loop
		"""
		# log bias (per neuron)
		log_bias = torch.zeros(
			(1, self.latent_dim),
			dtype=torch.float,
		)
		self.log_bias = nn.Parameter(
			data=log_bias - 3.0,
			requires_grad=True,
		)
		# log learning rate (global)
		self.log_lr = nn.ParameterList(nn.Parameter(
			data=torch.zeros(1),
			requires_grad=True,
		) for _ in range(self._cfg.len))
		return

	def _init_weight(self):
		if self._cfg.init_scale is None:
			return
		kws = dict(
			dist_name=self._cfg.init_dist,
			scale=self._cfg.init_scale,
			loc=0,
		)
		if self._cfg.init_dist == 't':
			kws['df'] = 2
		initializer = Initializer(**kws)
		initializer.apply(self.phi.weight)
		return

	def _init_buffers(self):
		self.register_buffer(
			name='temp',
			tensor=torch.tensor(1.0),
		)
		self.register_buffer(
			name='beta_inner',
			tensor=torch.tensor(self._cfg.beta_inner),
		)
		self.register_buffer(
			name='eps',
			tensor=torch.tensor(1.0),
		)
		return

	def _init_prior(self):
		raise NotImplementedError

	def __repr__(self):
		act_name = getattr(self._cfg, 'latent_act', None)
		act_name = '' if act_name is None else act_name
		name_str = [
			self._cfg.type.capitalize(),
			act_name.capitalize(),
			'Layer',
		]
		name_str = ''.join([s for s in name_str if s])
		dim_str = ', '.join([
			f"input_dim={self.input_dim}",
			f"latent_dim={self.latent_dim}",
		])
		main_info = ', '.join([
			f"{name_str}({dim_str}",
			f"temperature={self.temp.item():0.2g}",
			f"beta_inner={self.beta_inner.item():0.2g}",
			f"eps={self.eps.item():0.2g})",
		])
		return main_info


class PoissonLayer(_StochasticLayer):
	def __init__(
			self,
			input_dim: int,
			latent_dim: int,
			cfg: ConfigPoisVAE,
	):
		super(PoissonLayer, self).__init__(
			input_dim, latent_dim, cfg)
		self._init_n_exp()

	def grad_kl_term(self):
		return self.u[0] - self.u_prior[0]

	def grad_free_energy(self, x):
		grad_f = self.grad_recon_term(x)
		# kl? only when more than one iters
		if self._cfg.n_iters_inner > 1:
			grad_kl = self.grad_kl_term()
			grad_f = grad_f + grad_kl.mul(self.beta_inner)
		grad_f = grad_f + self.log_bias.exp()
		grad_f = (grad_f, )  # for consistency
		return grad_f

	def softclamp(self, x: torch.Tensor, which: str, *args):
		return softclamp_upper(
			x=x.clamp_(min=-8.0),
			c=self.get_c(which),
		)

	def get_dist(self, posterior: bool = True):
		# (1) get n_exp
		try:
			n_exp = self.n_exp[self.time]
		except IndexError:
			n_exp = 'infer'
		# (2) get dist
		dist = Poisson(
			log_rate=self.u[0] if posterior
			else self.u_prior[0],
			temp=self.temp.clone(),
			n_exp=n_exp if posterior
			else 'infer',
			n_exp_p=1e-3,
			clamp=None,
		)
		return dist

	def loss_kl(self):
		u = self.u[0]
		du = u - self.u_prior[0]
		f = 1 + torch.exp(du) * (du - 1)
		kl = torch.exp(u) * f
		return kl

	def update_n_exp(
			self,
			rates: Union[float, List[float]],
			p: float = 1e-3, ):
		if rates is None:
			return
		if not isinstance(rates, list):
			rates = [rates] * self.t_total
		assert all(rates) > 0.0, f"must be positive, got: {rates}"
		assert len(rates) == self.t_total
		# update values
		for i, r in enumerate(rates):
			self.n_exp[i] = compute_n_exp(r, p)
		return

	def _init_prior(self):
		rng = get_rng(self._cfg.seed)
		size = (1, self.latent_dim)
		kws = {'size': size}
		# u: membrane potentials
		if self._cfg.prior_log_dist == 'cte':
			u_init = np.ones(kws['size'])
			u_init *= self._cfg.clamp_prior
		elif self._cfg.prior_log_dist == 'uniform':
			kws.update(dict(low=-6.0, high=self._cfg.clamp_prior))
			u_init = rng.uniform(**kws)
		elif self._cfg.prior_log_dist == 'normal':
			s = np.abs(np.log(np.abs(self._cfg.clamp_prior)))
			kws.update(dict(loc=0.0, scale=s))
			u_init = rng.normal(**kws)
		else:
			raise NotImplementedError(self._cfg.prior_log_dist)
		u_init[u_init > 6.0] = 0.0
		u_init = nn.Parameter(
			data=torch.tensor(u_init, dtype=torch.float),
			requires_grad=self._cfg.fit_prior,
		)
		# finalize: as list
		self.u_init = nn.ParameterList([u_init])
		return

	def _init_n_exp(self):
		n_exp = torch.zeros(self.t_total, dtype=torch.int)
		self.register_buffer(name='n_exp', tensor=n_exp)
		self.update_n_exp(3.0)  # yields: n_exp = 10
		return


class GaussianLayer(_StochasticLayer):
	def __init__(
			self,
			input_dim: int,
			latent_dim: int,
			cfg: ConfigGausVAE,
	):
		super(GaussianLayer, self).__init__(
			input_dim, latent_dim, cfg)

	def grad_kl_term(self):
		# d/dmu kl
		delta_mu = self.u[0] - self.u_prior[0]
		var_prior = torch.exp(2 * self.u_prior[1])
		grad_mu = delta_mu / var_prior
		# d/dxi kl
		delta_xi = self.u[1] - self.u_prior[1]
		grad_xi = torch.exp(2 * delta_xi) - 1.0
		# put together
		grad_kl = (grad_mu, grad_xi)
		return grad_kl

	def grad_free_energy(self, x):
		# (1) recon
		grad_recon = self.grad_recon_term(x)
		# (2) retrieve noise (eps), compute sigma
		eps = self.posterior.retrieve_noise(self.samples)
		sigma = torch.exp(self.u[1])
		# (3) mu, xi
		grad_mu = grad_recon.clone()
		grad_xi = grad_recon * eps * sigma
		# (4) kl? only if more than one iters
		if self._cfg.n_iters_inner > 1:  # if just 1 iter: BONG
			grad_kl = self.grad_kl_term()
			grad_mu = grad_mu + grad_kl[0].mul(self.beta_inner)
			grad_xi = grad_xi + grad_kl[1].mul(self.beta_inner)
		# (5) apply inverse Fisher matrix
		grad_mu = grad_mu * sigma.pow(2)
		grad_xi = grad_xi * 0.5
		# (6) put together
		grad_f = (grad_mu + self.log_bias.exp(), grad_xi)
		return grad_f

	def softclamp(self, x: torch.Tensor, which: str, idx: int):
		return softclamp_sym(x, self.get_c(which, idx))

	def get_dist(self, posterior: bool = True):
		dist = Normal(
			loc=self.u[0] if posterior
			else self.u_prior[0],
			log_scale=self.u[1] if posterior
			else self.u_prior[1],
			temp=self.temp.clone(),
			clamp=None,
		)
		return dist

	def loss_kl(self):
		posterior = self.get_dist(posterior=True)
		prior = self.get_dist(posterior=False)
		kl = posterior.kl(prior)
		return kl

	def _init_prior(self):
		size = (1, self.latent_dim)
		# u: (u = membrane potential, xi = log sigma)
		self.u_init = nn.ParameterList(nn.Parameter(
			data=torch.zeros(size),
			requires_grad=self._cfg.fit_prior,
		) for _ in range(self._cfg.len))
		return


class LinearDictionary(nn.Module):
	def __init__(self, dim_1: int, dim_2: int):
		super(LinearDictionary, self).__init__()
		self.weight = nn.Parameter(
			data=torch.randn(dim_1, dim_2),
			requires_grad=True,
		)

	def forward(self, z, transpose=False):
		if transpose:
			return F.linear(z, self.weight.T)
		else:
			return F.linear(z, self.weight)
