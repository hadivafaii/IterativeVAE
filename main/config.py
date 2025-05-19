from base.config_base import *
from base.dataset import dataset_dims

_KL_BALANCER_CHOICES = [None, 'equal', 'linear', 'square', 'exp']
_KL_ADJUSTER_CHOICES = [None, 'equal', 'linear', 'sqrt', 'log']
_LOG_DIST_CHOICES = ['cte', 'uniform', 'normal']
_DEC_CHOICES = ['lin', 'mlp', 'conv', 'deconv']
_ENC_CHOICES = ['lin', 'mlp', 'conv', 'jacob']
_INF_CHOICES = ['grad', 'leapfrog']
T_ANNEAL_CHOICES = ['lin', 'exp']
METHOD_CHOICES = ['mc', 'exact']
DATA_CHOICES = [
	'vH16-bw', 'vH32-bw', 'Kyoto16-bw', 'Kyoto32-bw',
	'vH16-wht', 'vH32-wht', 'Kyoto16-wht', 'Kyoto32-wht',
	'vH16-col', 'vH32-col', 'Kyoto16-col', 'Kyoto32-col',
	'MNIST', 'FashionMNIST', 'EMNIST', 'Omniglot', 'SVHN',
	'ImageNet32', 'CelebA',
	'BALLS16', 'BALLS32', 'BALLS64',
	# datasets with feeders
	'vH32+jitter', 'Kyoto32+jitter',
	'MNIST+jitter', 'MNIST+rotate',
]


class ConfigVAE(BaseConfig):
	def __init__(
			self,
			dataset: str,
			t_train: int = 16,
			n_iters_outer: int = 1,
			n_iters_inner: int = 1,
			beta_outer: float = 16.0,
			beta_inner: float = 1.0,
			n_latents: int | Sequence[int] = 512,
			inf_type: str = 'grad',
			dec_type: str = 'lin',
			fit_prior: bool = True,
			stochastic: bool = True,
			init_dist: str = 'normal',
			init_scale: float = 0.05,
			clamp_du: float | Tuple[float, ...] = 7.0,
			clamp_u: float | Tuple[float, ...] = 8.0,
			kws_feeder: dict = None,
			**kwargs,
	):
		"""
		:param dataset: see above for allowed datasets
		:param t_train: sequence length (# train iterations)
		:param n_iters_outer: # repeats of the outer loop (tbpp)
		:param n_iters_inner: # gradient updates (during inference)
		:param beta_outer: beta used during learning
		:param beta_inner: beta used during inference
		:param n_latents: dimensionality of the latent space
		:param enc_type: {'grad'}
		:param dec_type: {'lin', 'conv'}
		:param init_dist: weight init distribution
		:param init_scale: weight init scale
		:param kwargs:
		"""
		assert dataset in DATA_CHOICES, \
			f"allowed datasets:\n{DATA_CHOICES}"
		assert dec_type in _DEC_CHOICES, \
			f"allowed decoders:\n{_DEC_CHOICES}"
		# assert enc_type in _ENC_CHOICES, \
		# 	f"allowed encoders:\n{_ENC_CHOICES}"
		assert inf_type in _INF_CHOICES, \
			f"allowed encoders:\n{_INF_CHOICES}"

		if isinstance(n_latents, int):
			n_latents = [n_latents]

		assert t_train >= 1
		assert n_iters_outer >= 1
		assert n_iters_inner >= 1

		self.t_train = int(t_train)
		self.n_iters_outer = int(n_iters_outer)
		self.n_iters_inner = int(n_iters_inner)

		self._setup_dataset(dataset)
		self._init_feeder_cfg(kws_feeder)
		self._init_input_sz()

		self.beta_inner = beta_inner
		self.beta_outer = beta_outer
		self.n_latents = n_latents
		self.inf_type = inf_type
		self.dec_type = dec_type

		self.clamp_u = clamp_u
		self.clamp_du = clamp_du
		self.init_dist = init_dist
		self.init_scale = init_scale
		self.stochastic = stochastic
		self.fit_prior = fit_prior

		super(ConfigVAE, self).__init__(**kwargs)

	@property
	def type(self):
		return NotImplemented

	@property
	def validation_temp(self):
		if self.type == 'poisson':
			return 0.0
		elif self.type in ['gaussian', 'laplace']:
			return 1.0
		else:
			raise ValueError(self.type)

	@property
	def len(self):
		if self.type == 'poisson':
			return 1
		elif self.type in ['gaussian', 'laplace']:
			return 2
		else:
			raise ValueError(self.type)

	@property
	def str_model(self):
		latent_name = getattr(
			self, 'latent_act', None)
		if latent_name is None:
			return self.type
		return '+'.join([self.type, latent_name])

	@property
	def str_t(self):
		str_t = f"t-{self.t_train}"
		if self.n_iters_outer > 1:
			str_t = f"{str_t}×{self.n_iters_outer}"
		if self.n_iters_inner > 1:
			str_t = f"{str_t}[×{self.n_iters_inner}]"
		return str_t

	@property
	def str_b(self):
		str_b = f"b-{self.beta_outer:0.4g}"
		if self.n_iters_inner > 1:
			str_b = f"{str_b}×{self.beta_inner:0.4g}"
		return str_b

	@property
	def str_k(self):
		str_k = ','.join([
			str(k) for k in
			self.n_latents
		])
		return f"k-[{str_k}]"

	@property
	def str_c(self):
		str_c = ','.join([
			f"u-{_clamp_str(self.clamp_u)}",
			f"du-{_clamp_str(self.clamp_du)}",
		]).replace(' ', '')
		return str_c

	def _setup_dataset(self, dataset):
		self.dataset = dataset
		parts = dataset.partition('+')
		self.dataset_name, _, feeder = parts
		self.feeder = feeder or 'stationary'
		return

	def _init_input_sz(self):
		c, h, w = dataset_dims(self.dataset_name)
		# used for: vH32-jitter + crop to 16
		crop_size = self.feeder_cfg.get('crop_size')
		if crop_size is not None:
			h, w = crop_size, crop_size
		self.input_sz = (c, h, w)
		self.shape = (-1, *self.input_sz)
		return

	def _init_feeder_cfg(self, kws_feeder):
		defaults = default_feeder_configs(
			self.dataset_name, self.feeder)
		self.feeder_cfg = setup_kwargs(
			defaults, kws_feeder)
		return

	def _name(self):
		name = [
			self.str_model,
			str(self.dataset),
			self.str_t,
			self.str_b,
			self.str_k,
			self.str_c,
		]
		# prior is not fit?
		if not self.fit_prior:
			name += ['fixed-prior']
		# add achitecture specs
		# not necessary for now
		# name += [self.attr2archi()]
		# filter and join
		name = '_'.join([s for s in name if s])
		return name

	def attr2archi(self) -> str:
		archi = '|'.join([
			f"<{self.inf_type}",
			f"{self.dec_type}>",
		])
		return archi

	def save(self):
		kws = dict(
			with_base=True,
			overwrite=True,
			verbose=False,
		)
		self._save(**kws)


class ConfigPoisVAE(ConfigVAE):
	def __init__(
			self,
			clamp_prior: float = -2.0,
			prior_log_dist: str = 'uniform',
			**kwargs,
	):
		assert prior_log_dist in _LOG_DIST_CHOICES, \
			f"allowed prior log_dists:\n{_LOG_DIST_CHOICES}"
		self.prior_log_dist = prior_log_dist
		self.clamp_prior = clamp_prior
		super(ConfigPoisVAE, self).__init__(
			**kwargs)

	@property
	def type(self):
		return 'poisson'


class _ConfigContVAE(ConfigVAE):
	def __init__(
			self,
			model_type: str,
			latent_act: str = None,
			**kwargs,
	):
		self._type = model_type
		self.latent_act = latent_act
		super(_ConfigContVAE, self).__init__(
			**kwargs)

	@property
	def type(self):
		return self._type


class ConfigGausVAE(_ConfigContVAE):
	def __init__(self, **kwargs):
		super(ConfigGausVAE, self).__init__(
			model_type='gaussian', **kwargs)


class ConfigLapVAE(_ConfigContVAE):
	def __init__(self, **kwargs):
		super(ConfigLapVAE, self).__init__(
			model_type='laplace', **kwargs)


class ConfigTrain(BaseConfigTrain):
	def __init__(
			self,
			kl_beta_min: float = 1e-4,
			kl_balancer: str = None,
			kl_time_adjuster: str = None,
			kl_anneal_cycles: int = 0,
			kl_anneal_portion: float = 0.1,
			kl_const_portion: float = 1e-3,
			temp_anneal_portion: float = 0.5,
			temp_anneal_type: str = 'lin',
			temp_start: float = 1.0,
			temp_stop: float = 0.01,
			**kwargs,
	):
		"""
		:param kl_beta_min:
		:param kl_balancer:
		:param kl_time_adjuster:
		:param kl_anneal_cycles:
		:param kl_anneal_portion:
		:param kl_const_portion:
		:param temp_anneal_portion:
		:param temp_anneal_type:
		:param temp_start:
		:param temp_stop:
		:param kwargs:
		"""
		defaults = dict(
			lr=0.002,
			epochs=1200,
			batch_size=200,
			warm_restart=0,
			warmup_portion=0.005,
			optimizer='adamax_fast',
			optimizer_kws={
				'weight_decay': 3e-4},
			scheduler_type='cosine',
			grad_clip=500,
			chkpt_freq=50,
			eval_freq=20,
			log_freq=10,
		)
		kwargs = setup_kwargs(defaults, kwargs)
		super(ConfigTrain, self).__init__(**kwargs)
		self.set_scheduler_kws()  # reset scheduler kws
		assert 0.0 <= kl_anneal_portion <= 1.0
		assert 0.0 <= temp_anneal_portion <= 1.0
		assert temp_anneal_type in T_ANNEAL_CHOICES, \
			f"allowed t annealers:\n{T_ANNEAL_CHOICES}"
		assert kl_balancer in _KL_BALANCER_CHOICES, \
			f"allowed kl balancers:\n{_KL_BALANCER_CHOICES}"
		assert kl_time_adjuster in _KL_ADJUSTER_CHOICES, \
			f"allowed kl adjusters:\n{_KL_ADJUSTER_CHOICES}"
		# kl
		self.kl_beta_min = kl_beta_min
		self.kl_balancer = kl_balancer
		self.kl_time_adjuster = kl_time_adjuster
		self.kl_anneal_cycles = kl_anneal_cycles
		self.kl_anneal_portion = kl_anneal_portion
		self.kl_const_portion = kl_const_portion
		# temp
		self.temp_anneal_portion = temp_anneal_portion
		self.temp_anneal_type = temp_anneal_type
		self.temp_start = temp_start
		self.temp_stop = temp_stop

	# def str_b(self, beta_inner: float = 1.0):
	# 	str_b = f"b-{self.kl_beta:0.4g}"
	# 	if beta_inner > 1:
	# 		str_b = f"{str_b}×{beta_inner:0.4g}"
	# 	return str_b

	def name(self):
		# str_beta = [f"beta({self.kl_beta:0.4g}:"]
		# if self.kl_anneal_cycles > 0:
		# 	str_beta.append(f"{self.kl_anneal_cycles}x")
		# str_beta.append(f"{self.kl_anneal_portion:0.1g})")
		name = [
			'-'.join([
				f"b{self.batch_size}",
				f"ep{self.epochs}",
				f"lr({self.lr:0.2g})",
			]),
			# ''.join(str_beta),
		]
		if self.temp_anneal_portion > 0:
			str_temp = [
				'temp',
				f"({self.temp_stop:0.2g}:",
			]
			if self.temp_anneal_type != 'lin':
				str_temp.append(f"{self.temp_anneal_type}-")
			str_temp.append(f"{self.temp_anneal_portion:0.1g})")
			name.append(''.join(str_temp))
		if self.grad_clip is not None:
			name.append(f"gr({self.grad_clip})")
		return '_'.join(name)

	def save(self, save_dir: str):
		kws = dict(
			with_base=True,
			overwrite=True,
			verbose=False,
		)
		self._save(save_dir, **kws)


def default_feeder_configs(
		dataset: str,
		feeder: str, ):
	# defaults
	config_map = {
		('MNIST', 'SVHN'): {
			'jitter': dict(
				mode='pad',
				diff_coeff=0.6,
				harmonic_strength=0.03),
			'rotate': dict(
				theta_0=None,
				delta_theta=8.0),
			'stationary': dict(),
		},
		(
			'vH', 'Kyoto', 'BALLS',
			'ImageNet32', 'CelebA'
		): {
			'jitter': dict(
				mode='crop',
				crop_size=16,
				diff_coeff=0.5,
				harmonic_strength=0.01),
			'stationary': dict(),
		}
	}
	# Find matching dataset
	for key_tuple, cfg in config_map.items():
		for key in key_tuple:
			if key in dataset:
				if feeder in cfg:
					return cfg[feeder]
				msg = f"'{feeder}' / '{dataset}'"
				raise NotImplementedError(msg)
	raise NotImplementedError(dataset)


def default_configs(
		model_type: str,
		dataset: str = 'vH16-wht',
		archi_type: str = 'grad|lin', ):

	########################
	# vae —— model specific
	########################
	if model_type == 'poisson':
		cfg_mod = dict(
			clamp_prior=-2,
			clamp_du=7.0,
			clamp_u=8.0,
		)
		cfg_tr = dict()
	elif model_type in ['gaussian', 'laplace']:
		cfg_mod = dict(
			clamp_du=(7.0, 3.0),
			clamp_u=(7.0, 2.0),
		)
		cfg_tr = dict(
			temp_anneal_portion=0.0,
			temp_stop=1.0,
		)
	else:
		raise ValueError(model_type)

	# finalize vae cfg
	cfg_mod = {**cfg_mod, **_archi2attr(archi_type)}
	if cfg_mod['dec_type'] == 'lin':
		if 'clamp_prior' in cfg_mod:
			cfg_mod['clamp_u'] = 10.0
			cfg_mod['clamp_du'] = 10.0
			cfg_mod['clamp_prior'] = -4
		n_latents = 512
	elif cfg_mod['dec_type'] == 'mlp':
		n_latents = 128
	elif cfg_mod['dec_type'] == 'conv':
		n_latents = 64
	else:
		n_latents = 32

	cfg_mod = {
		'dataset': dataset,
		'n_latents': n_latents,
		**cfg_mod,
	}
	# init: dist & scale
	cfg_mod.update(dict(
		init_dist='normal',
		init_scale=1e-4 if
		cfg_mod['dec_type'] == 'lin'
		else 0.05,
	))

	########################
	# trainer cfgs
	########################
	if any(s in dataset for s in ['vH', 'Kyoto', 'CIFAR16']):
		cfg_mod['n_latents'] = 512
		match dataset.split('-')[0]:
			case 'vH16': epochs = 300
			case 'Kyoto16': epochs = 800
			case 'CIFAR16': epochs = 200
			case _: epochs = 300
		cfg_tr = dict(
			**cfg_tr,
			batch_size=200,
			epochs=epochs,
			grad_clip=300,
			lr=0.005,
		)

	elif dataset in ['MNIST', 'FashionMNIST', 'EMNIST', 'Omniglot', 'SVHN']:
		# grad_clip = 50 if cfg_mod['dec_type'] == 'mlp' else 150
		grad_clip = 3000
		chkpt_freq = 20
		epochs = 400
		if dataset == 'Omniglot':
			chkpt_freq = 50
			epochs = 1250
		elif dataset == 'EMNIST':
			epochs = 200
		elif dataset == 'SVHN':
			grad_clip = 10_000
			epochs = 300
		cfg_tr = dict(
			**cfg_tr,
			epochs=epochs,
			batch_size=100,
			# warm_restart=1,
			chkpt_freq=chkpt_freq,
			grad_clip=grad_clip,
			lr=0.0005 if dataset in ['SVHN', 'FashionMNIST']
			else 0.002,
		)

	# elif dataset == 'SVHN':
	# 	cfg_mod['init_scale'] = 1e-3
	# 	cfg_mod['n_latents'] = 256
	# 	cfg_tr = dict(
	# 		**cfg_tr,
	# 		epochs=350,
	# 		batch_size=200,
	# 		# warm_restart=1,
	# 		grad_clip=100 if cfg_mod['dec_type'] == 'mlp' else 300,
	# 	)

	elif dataset in ['CIFAR10', 'ImageNet32', 'CelebA']:
		# cfg_mod['clamp_u'] = 6.5
		# cfg_mod['clamp_du'] = 6.0
		# if 'clamp_prior' in cfg_mod:
		# 	cfg_mod['clamp_prior'] = 1.0
		# cfg_mod['init_scale'] = 1e-3
		cfg_mod['n_latents'] = 768
		epochs = 1000
		batch_size = 500
		if dataset == 'ImageNet32':
			epochs = 120
		if dataset == 'CelebA':
			batch_size = 100
			epochs = 200
		cfg_tr = dict(
			**cfg_tr,
			epochs=epochs,
			batch_size=batch_size,
			# warm_restart=0,
			# optimizer_kws={'weight_decay': 1e-3},
			scheduler_kws={'eta_min': 1e-6},
			grad_clip=100,  # if model_type == 'gaussian' else 4000,
			chkpt_freq=10,  # if dataset == 'CIFAR10' else 1,
			eval_freq=10,  # if dataset == 'CIFAR10' else 1,
			lr=2e-3,
		)

	elif dataset.startswith('BALLS'):
		cfg_tr = dict(
			**cfg_tr,
			epochs=600,
			batch_size=200,
			grad_clip=5000,
			lr=0.0005,
		)

	else:
		raise ValueError(dataset)

	# finalize trainer cfg
	# cfg_tr['kl_const_portion'] = 0.0 \
	# 	if linear_decoder else 0.01

	return cfg_mod, cfg_tr


def _archi2attr(architecture: str):
	# get rid of the <bra|ket>
	if architecture.startswith('<'):
		architecture = architecture[1:]
	if architecture.endswith('>'):
		architecture = architecture[:-1]
	inf, dec = architecture.split('|')
	# attrs
	archi_attrs = dict(
		inf_type=str(inf),
		dec_type=str(dec),
	)
	return archi_attrs


def _clamp_str(clmap):
	if isinstance(clmap, float):
		str_u = f"{clmap:0.3g}"
	else:
		str_u = ','.join(
			f"{c:0.3g}"
			for c in clmap
		)
		str_u = f"({str_u})"
	return str_u


CFG_CLASSES = {
	'poisson': ConfigPoisVAE,
	'gaussian': ConfigGausVAE,
	'laplace': ConfigLapVAE,
}
