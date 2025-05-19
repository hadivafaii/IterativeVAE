# noinspection PyUnresolvedReferences
from utils.plotting import *
from main.config import *
from torch import distributions as dists


def kl_balancer(
		kl_batch: torch.Tensor,
		alpha: torch.Tensor = None,
		balance: bool = True,
		eps: float = 0.01, ):
	"""
	kl_batch: [B, T]
	alpha: [T], balancer coeffs
	gamma: [T], sum(gamma) = T
	"""
	gamma = torch.mean(
		kl_batch.detach().abs(),
		keepdim=True,
		dim=0,
	) + eps
	if alpha is not None:
		gamma *= alpha.unsqueeze(0)
	gamma /= torch.mean(gamma, keepdim=True, dim=1)
	if balance:
		kl_balanced = torch.mean(kl_batch * gamma, dim=1)
	else:
		kl_balanced = torch.mean(kl_batch, dim=1)
	return kl_balanced, gamma.squeeze(0)


def kl_balancer_coeff(
		t: int,
		fun: str,
		flip: bool = True,
		normalize: bool = False, ):
	if fun == 'equal':
		coeff = torch.ones(t)
	elif fun == 'linear':
		coeff = torch.arange(1, t + 1)
	elif fun == 'sqrt':
		coeff = torch.arange(1, t + 1) ** 0.5
	elif fun == 'square':
		coeff = torch.arange(1, t + 1) ** 2
	elif fun == 'exp':
		coeff = torch.arange(1, t + 1).exp()
	elif fun == 'log':
		coeff = torch.arange(1, t + 1).log() + 1
	else:
		raise NotImplementedError(fun)
	coeff = coeff.float()
	coeff /= torch.min(coeff)
	if flip:
		coeff = torch.flip(
			coeff, dims=[0])
	if normalize:
		coeff /= coeff.mean()
	return coeff


def temp_anneal_linear(
		n_iters: int,
		t0: float = 1.0,
		t1: float = 0.1,
		portion: float = 0.7, ):
	temperatures = np.ones(n_iters) * t1
	i = int(np.ceil(portion * n_iters))
	temperatures[:i] = np.linspace(t0, t1, i)
	return temperatures


def temp_anneal_exp(
		n_iters: int,
		t0: float = 1.0,
		t1: float = 0.1,
		portion: float = 0.7,
		rate: float = 'infer', ):
	n = int(np.ceil(n_iters * portion))
	n = min(n, n_iters)
	if rate == 'infer':
		rate = - (n/(n-1)) * np.log(t1/100)
	else:
		assert isinstance(rate, float)
	temperatures = np.ones(n_iters) * t1
	for i in range(n):
		coef = np.exp(-rate * i / n)
		t = t1 + (t0 - t1) * coef
		temperatures[i] = t
	return temperatures


def anneal_cosine(
		n_iters: int,
		start: float = 0.0,
		stop: float = 1.0,
		n_cycles: int = 4,
		portion: float = 0.5,
		final_value: float = 1.0, ):
	period = n_iters / n_cycles
	step = (stop-start) / (period*portion)
	betas = np.ones(n_iters) * final_value
	for c in range(n_cycles):
		v, i = start, 0
		while v <= stop:
			val = (1 - np.cos(v*np.pi)) * final_value / 2
			betas[int(i+c*period)] = val
			v += step
			i += 1
	return betas


def anneal_linear(
		n_iters: int,
		final_value: float = 1.0,
		anneal_portion: float = 0.3,
		constant_portion: float = 0,
		min_value: float = 1e-4, ):
	betas = np.ones(n_iters) * final_value
	a = int(np.ceil(constant_portion * n_iters))
	b = int(np.ceil((constant_portion + anneal_portion) * n_iters))
	betas[:a] = min_value
	betas[a:b] = np.linspace(min_value, final_value, b - a)
	return betas


def compute_r2(true, pred):
	ss_res = (true - pred).pow(2)
	ss_tot = (true - true.mean(
		dim=-1, keepdim=True)).pow(2)
	ss_res = torch.sum(ss_res, dim=-1)
	ss_tot = torch.sum(ss_tot, dim=-1)
	# compute r2
	eps = torch.finfo(torch.float32).eps
	r2 = 1.0 - ss_res / (ss_tot + eps)
	return r2


def load_model(
		s: str,
		device: str,
		lite: bool = False,
		**kwargs, ):
	kwargs.setdefault('device', device)
	if lite:
		tr, meta = load_model_lite(s, **kwargs)
	else:
		tr, meta = load_model_main(*s.split('/'), **kwargs)
	return tr, meta


def load_model_lite(
		path: str,
		device: str = 'cpu',
		strict: bool = True,
		verbose: bool = False,
		**kwargs, ):
	# load model
	cfg = next(
		e for e in os.listdir(path)
		if e.startswith('Config')
		and e.endswith('.json')
		and 'Train' not in e
	)
	fname = cfg.split('.')[0]
	cfg = pjoin(path, cfg)
	with open(cfg, 'r') as f:
		cfg = json.load(f)
	# extract key
	key = next(
		k for k, cls in
		CFG_CLASSES.items()
		if fname == cls.__name__
	)
	# load cfg/model
	cfg = CFG_CLASSES[key](save=False, **cfg)
	from main.model import MODEL_CLASSES
	model = MODEL_CLASSES[key](
		cfg, verbose=verbose)

	# load state dict
	fname_pt = next(
		f for f in os.listdir(path)
		if f.split('.')[-1] == 'pt'
	)
	state_dict = pjoin(path, fname_pt)
	state_dict = torch.load(
		f=state_dict,
		map_location='cpu',
		weights_only=False,  # TODO: later revert to True
	)
	ema = state_dict['model_ema'] is not None
	model.load_state_dict(
		state_dict=state_dict['model'],
		strict=strict,
	)

	# set chkpt_dir & timestamp
	model.chkpt_dir = path
	timestamp = state_dict['metadata'].get('timestamp')
	if timestamp is not None:
		model.timestamp = timestamp

	# load trainer
	cfg_train = next(
		e for e in os.listdir(path)
		if e.startswith('Config')
		and e.endswith('.json')
		and 'Train' in e
	)
	fname = cfg_train.split('.')[0]
	cfg_train = pjoin(path, cfg_train)
	with open(cfg_train, 'r') as f:
		cfg_train = json.load(f)
	if fname.endswith('Train'):
		from main.train import Trainer
		cfg_train = ConfigTrain(**cfg_train)
		kwargs.setdefault('shuffle', False)
		trainer = Trainer(
			model=model,
			cfg=cfg_train,
			device=device,
			verbose=verbose,
			**kwargs,
		)
	else:
		raise NotImplementedError

	if ema:
		trainer.model_ema.load_state_dict(
			state_dict=state_dict['model_ema'],
			strict=strict,
		)
		if timestamp is not None:
			trainer.model_ema.timestamp = timestamp

	# optim, etc.
	if strict:
		trainer.optim.load_state_dict(
			state_dict['optim'])
		if trainer.optim_schedule is not None:
			trainer.optim_schedule.load_state_dict(
				state_dict.get('scheduler', {}))
	# stats
	stats_model = state_dict['metadata'].pop('stats_model', {})
	stats_trainer = state_dict['metadata'].pop('stats_trainer', {})
	trainer.model.stats.update(stats_model)
	trainer.stats.update(stats_trainer)
	# meta
	metadata = {
		**state_dict['metadata'],
		'file': fname_pt,
	}
	return trainer, metadata


def load_model_main(
		model_name: str,
		fit_name: Union[str, int] = -1,
		checkpoint: int = -1,
		device: str = 'cpu',
		strict: bool = True,
		verbose: bool = False,
		path: str = 'Projects/PoissonVAE/models',
		**kwargs, ):
	# cfg model
	path = pjoin(add_home(path), model_name)
	fname = next(s for s in os.listdir(path) if 'json' in s)
	with open(pjoin(path, fname), 'r') as f:
		cfg = json.load(f)
	# extract key
	fname = fname.split('.')[0]
	key = next(
		k for k, cls in
		CFG_CLASSES.items()
		if fname == cls.__name__
	)
	# load cfg/model
	cfg = CFG_CLASSES[key](save=False, **cfg)
	from main.model import MODEL_CLASSES
	model = MODEL_CLASSES[key](
		cfg, verbose=verbose)

	# now enter the fit folder
	if isinstance(fit_name, str):
		path = pjoin(path, fit_name)
	elif isinstance(fit_name, int):
		path = sorted(filter(
			os.path.isdir, [
				pjoin(path, e) for e
				in os.listdir(path)
			]
		), key=_sort_fn)[fit_name]
	else:
		raise ValueError(fit_name)
	files = sorted(os.listdir(path))

	# load state dict
	fname_pt = [
		f for f in files if
		f.split('.')[-1] == 'pt'
	]
	if checkpoint == -1:
		fname_pt = fname_pt[-1]
	else:
		fname_pt = next(
			f for f in fname_pt if
			checkpoint == _chkpt(f)
		)
	state_dict = pjoin(path, fname_pt)
	state_dict = torch.load(
		f=state_dict,
		map_location='cpu',
		weights_only=False,  # TODO: later revert to True
	)
	ema = state_dict['model_ema'] is not None
	model.load_state_dict(
		state_dict=state_dict['model'],
		strict=strict,
	)

	# set chkpt_dir & timestamp
	model.chkpt_dir = path
	timestamp = state_dict['metadata'].get('timestamp')
	if timestamp is not None:
		model.timestamp = timestamp

	# load trainer
	fname = next(
		f for f in files if
		f.split('.')[-1] == 'json'
	)
	with open(pjoin(path, fname), 'r') as f:
		cfg_train = json.load(f)
	fname = fname.split('.')[0]
	if fname == 'ConfigTrain':
		from main.train import Trainer
		cfg_train = ConfigTrain(**cfg_train)
		kwargs.setdefault('shuffle', False)
		trainer = Trainer(
			model=model,
			cfg=cfg_train,
			device=device,
			verbose=verbose,
			**kwargs,
		)
	else:
		raise NotImplementedError(fname)

	if ema:
		trainer.model_ema.load_state_dict(
			state_dict=state_dict['model_ema'],
			strict=strict,
		)
		if timestamp is not None:
			trainer.model_ema.timestamp = timestamp

	if strict:
		trainer.optim.load_state_dict(
			state_dict['optim'])
		if trainer.optim_schedule is not None:
			trainer.optim_schedule.load_state_dict(
				state_dict.get('scheduler', {}))
	# stats
	stats_model = state_dict['metadata'].pop('stats_model', {})
	stats_trainer = state_dict['metadata'].pop('stats_trainer', {})
	trainer.model.stats.update(stats_model)
	trainer.stats.update(stats_trainer)
	# meta
	metadata = {
		**state_dict['metadata'],
		'file': fname_pt,
	}
	return trainer, metadata


def print_grad_table(
		trainer,
		metadata: dict,
		clip: float = None,
		thresholds: List[float] = None, ):
	thresholds = thresholds if thresholds else [
		1, 2, 5, 10, 20, 50, 100, 200]
	clip = clip if clip else trainer.cfg.grad_clip
	thresholds = [
		clip * i for i
		in thresholds
	]
	bad = np.array(list(trainer.stats['grad'].values()))

	t = PrettyTable(['Threshold', '#', '%'])
	for thres in thresholds:
		tot = (bad > thres).sum()
		perc = tot / metadata['global_step']
		perc = np.round(100 * perc, 3)
		t.add_row([int(thres), tot, perc])
	print(t, '\n')
	return


def print_num_params(
		module: nn.Module,
		full: bool = True, ):
	def _tot_params(_m):
		return sum(
			p.numel() for p
			in _m.parameters()
			if p.requires_grad
		)

	def _add_module(name, _m):
		tot = _tot_params(_m)
		if tot == 0:
			return

		if tot >= 1e6:
			tot = f"{np.round(tot / 1e6, 2):1.1f} Mil"
		elif tot >= 1e3:
			tot = f"{np.round(tot / 1e3, 2):1.1f} K"
		else:
			tot = str(tot)
		t.add_row([name, tot])
		return

	def _process_module(_prefix, _m):
		if isinstance(_m, (nn.ModuleDict, nn.ModuleList)):
			for _name, sub_module in _m.named_children():
				full_name = f"{_prefix}.{_name}" \
					if _prefix else _name
				_process_module(full_name, sub_module)
		else:
			_add_module(_prefix, _m)

	# top row: the full module
	t = PrettyTable(['Module Name', 'Num Params'])
	_add_module(module.__class__.__name__, module)
	t.add_row(['———', '———'])

	for prefix, m in module.named_modules():
		if '.' in prefix or not prefix:
			continue
		if full:
			_process_module(prefix, m)
		else:
			_add_module(prefix, m)

	print(t, '\n')
	return


def add_weight_decay(
		model: nn.Module,
		weight_decay: float = 1e-2,
		skip: Tuple[str, ...] = ('bias',), ):
	decay = []
	no_decay = []
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		if len(param.shape) <= 1 or any(k in name for k in skip):
			no_decay.append(param)
		else:
			decay.append(param)
	param_groups = [
		{'params': no_decay, 'weight_decay': 0.},
		{'params': decay, 'weight_decay': weight_decay},
	]
	return param_groups


class AverageMeter(object):
	def __init__(self):
		self.val = 0
		self.sum = 0
		self.avg = 0
		self.cnt = 0

	def reset(self):
		self.val = 0
		self.sum = 0
		self.avg = 0
		self.cnt = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.cnt += n
		self.avg = self.sum / self.cnt


class AverageMeterDict(object):
	def __init__(self, keys: Sequence[Any]):
		self.meters = {
			k: AverageMeter()
			for k in keys
		}

	def __getitem__(self, key):
		return self.meters.get(key)

	def __iter__(self):
		for key in self.meters:
			yield key

	def reset(self):
		for k, m in self.meters.items():
			m.reset()

	def update(self, vals: dict, n: int = 1):
		for k, m in self.meters.items():
			m.update(vals.get(k, 0), n)

	def avg(self):
		return {
			k: m.avg for k, m in
			self.meters.items()
		}


class Timer:
	def __init__(self):
		self.times = [time.time()]
		self.messages = {}
		self.counts = {}

	def __call__(self, message=None):
		self.times.append(time.time())
		t = self.times[-1] - self.times[-2]
		if message in self.messages:
			self.messages[message] += t
			self.counts[message] += 1
		else:
			self.messages[message] = t
			self.counts[message] = 1

	def print(self):
		for k, t in self.messages.items():
			print(f"{k}. Time: {t:0.1f}")


class Initializer:
	def __init__(self, dist_name: str, **kwargs):
		self.mode = 'pytorch'
		try:
			dist_module = getattr(dists, dist_name.lower())
			dist = getattr(dist_module, dist_name.title())
			kwargs = filter_kwargs(dist, kwargs)
		except AttributeError:
			dist = getattr(sp_stats, dist_name)
			self.mode = 'scipy'
		self.dist = dist(**kwargs)

	@torch.no_grad()
	def apply(self, weight: torch.Tensor):
		if self.mode == 'pytorch':
			values = self.dist.sample(weight.shape)
		else:
			values = self.dist.rvs(tuple(weight.shape))
			values = torch.tensor(values, dtype=torch.float)
		weight.data.copy_(values.to(weight.device))
		return


class Module(nn.Module):
	def __init__(self, cfg, verbose: bool = False):
		super(Module, self).__init__()
		self.chkpt_dir = None
		self.timestamp = now(True)
		self.stats = collections.defaultdict(list)
		self.verbose = verbose
		self.cfg = cfg

	def print(self):
		print_num_params(self)

	def create_chkpt_dir(self, fit_name: str = None):
		chkpt_dir = '_'.join([
			fit_name if fit_name else
			f"seed-{self.cfg.seed}",
			f"({self.timestamp})",
		])
		chkpt_dir = pjoin(
			self.cfg.mods_dir,
			chkpt_dir,
		)
		os.makedirs(chkpt_dir, exist_ok=True)
		self.chkpt_dir = chkpt_dir
		return

	def save(
			self,
			checkpoint: int = -1,
			name: str = None,
			path: str = None, ):
		path = path if path else self.chkpt_dir
		name = name if name else type(self).__name__
		fname = '-'.join([
			name,
			f"{checkpoint:04d}",
			f"({now(True)}).pt",
		])
		fname = pjoin(path, fname)
		torch.save(self.state_dict(), fname)
		return fname


def _chkpt(f):
	return int(f.split('_')[0].split('-')[-1])


def _sort_fn(f: str):
	f = f.split('(')[-1].split(')')[0]
	ymd, hm = f.split(',')
	yy, mm, dd = ymd.split('_')
	h, m = hm.split(':')
	yy, mm, dd, h, m = map(
		lambda s: int(s),
		[yy, mm, dd, h, m],
	)
	x = (
		yy * 1e8 +
		mm * 1e6 +
		dd * 1e4 +
		h * 1e2 +
		m
	)
	return x
