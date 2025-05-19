from .layers import *
from figures.imgs import plot_weights


class _IterativeVAE(Module):
	def __init__(self, cfg: ConfigVAE, **kwargs):
		super(_IterativeVAE, self).__init__(
			cfg, **kwargs)

	def forward(
			self,
			x: torch.Tensor,
			temp: float = None,
			return_extras: List[str] = None, ):
		# (1) infer
		_ = self.layer(x, temp=temp)
		# (2) generate:
		self.layer.generate(temp=temp)
		# (3) save values (main)
		output = {
			'loss_kl': self.layer.loss_kl(),
			'loss_recon': self.layer.loss_recon(x),
			'posterior': self.layer.posterior,
		}
		# (4) save values (extras?)
		if return_extras:
			output = self._add_extras(
				x, output, return_extras)
		# (5) track stats?
		if self.cfg.track_stats:
			self._track_states()
		return output

	@torch.no_grad()
	def xtract_ftr(self, x, **kwargs):
		temp = self.cfg.validation_temp
		kwargs.setdefault('temp', temp)
		return self.forward(x, **kwargs)

	@torch.no_grad()
	def generate(self, x, **kwargs):
		temp = self.cfg.validation_temp
		items = ['u', 'samples', 'recon']
		kwargs.setdefault('temp', temp)
		kwargs.setdefault('return_extras', items)
		kwargs.setdefault('override_input', True)
		return self.forward(x, **kwargs)

	def apply_act_fn(self, state: torch.Tensor):
		if self.cfg.type == 'poisson':
			return torch.exp(state)
		elif self.cfg.type == 'gaussian':
			return self.layer.apply_act_fn(state)
		else:
			raise NotImplementedError(self.cfg.type)

	def reset_inference_state(self, batch_size: int):
		self.layer.reset_state(batch_size)
		return

	def detach_inference_state(self):
		self.layer.detach_state()
		self.layer.attach_prior()  # TODO: expt
		return

	def update_time(self, t: int):
		self.layer.time = t
		return

	def update_temp(self, new_t: float):
		self.layer.update_temp(new_t)
		return

	def update_beta(self, new_beta: float):
		self.layer.update_beta(new_beta)
		return

	def update_eps(self, new_eps: float):
		self.layer.update_eps(new_eps)
		return

	@property
	def temp(self):
		return self.layer.temp

	@torch.no_grad()
	def _add_extras(
			self,
			x: torch.Tensor,
			output: Dict[str, torch.Tensor],
			items: List[str] = None, ):
		if isinstance(items, str):
			items = [items]
		elif isinstance(items, list):
			pass
		else:
			items = [
				'du', 'u', 'samples',
				'r2', 'mse',
			]
		if 'du' in items:
			du_tuple = tuple(
				u - u_prior for u, u_prior in
				zip(self.layer.u, self.layer.u_prior)
			)
			output['du'] = du_tuple[0].detach()
		if 'u' in items:
			output['u'] = self.layer.u[0].detach()
		if 'samples' in items:
			output['samples'] = self.layer.apply_act_fn().detach()
		if 'conditional' in items:
			output['conditional'] = self.layer.conditional
		if 'recon' in items:
			output['recon'] = self.layer.conditional.loc.detach()
			# output['recon'] = self.layer.conditional.sample().detach()
		if 'r2' in items:
			output['r2'] = self.layer.loss_recon(x, mode='r2')
		if 'mse' in items:
			output['mse'] = self.layer.loss_recon(x, mode='mse')
		if 'x' in items:
			output['x'] = x
		return output

	@torch.no_grad()
	def _track_states(self):
		du_tuple = tuple(
			u - u_prior for u, u_prior in
			zip(self.layer.u, self.layer.u_prior)
		)
		self.stats['u_min'].append(tuple(
			u.min().item() for u in self.layer.u
		))
		self.stats['u_max'].append(tuple(
			u.max().item() for u in self.layer.u
		))
		self.stats['du_min'].append(tuple(
			du.min().item() for du in du_tuple
		))
		self.stats['du_max'].append(tuple(
			du.max().item() for du in du_tuple
		))
		return

	def show(
			self,
			which: str = 'phi',
			order: Iterable[int] = None,
			method: str = 'min-max',
			add_title: bool = False,
			display: bool = True,
			**kwargs, ):
		assert which in ['phi']  # TODO
		# get weights
		w = self.layer.get_weight()
		w = w.T.reshape(self.cfg.shape)
		w = tonp(w.squeeze())
		if order is not None:
			w = w[order]

		pad = 0 if self.cfg.input_sz[0] == 3 else 1
		kwargs.setdefault('pad', pad)
		kwargs.setdefault('dpi', 200)
		fig, ax = plot_weights(
			w=w,
			method=method,
			title=None if not add_title else
			which.capitalize() + 'oder',
			display=display,
			**kwargs,
		)
		return fig, ax


class IPVAE(_IterativeVAE):
	def __init__(self, cfg: ConfigPoisVAE, **kwargs):
		super(IPVAE, self).__init__(cfg, **kwargs)
		self.layer = PoissonLayer(
			latent_dim=cfg.n_latents[0],
			input_dim=np.prod(cfg.input_sz),
			cfg=cfg,
		)


class IGVAE(_IterativeVAE):
	def __init__(self, cfg: ConfigGausVAE, **kwargs):
		super(IGVAE, self).__init__(cfg, **kwargs)
		self.layer = GaussianLayer(
			latent_dim=cfg.n_latents[0],
			input_dim=np.prod(cfg.input_sz),
			cfg=cfg,
		)


MODEL_CLASSES = {
	'poisson': IPVAE,
	'gaussian': IGVAE,
}
