from base.utils_model import *
from torch.utils.tensorboard import SummaryWriter


class BaseTrainer(object):
	def __init__(
			self,
			model: Module,
			cfg: Any,
			device: str = 'cpu',
			shuffle: bool = True,
			verbose: bool = False,
	):
		super(BaseTrainer, self).__init__()
		self.cfg = cfg
		self.shuffle = shuffle
		self.verbose = verbose
		self.device = torch.device(device)
		self.model = model.to(self.device).eval()
		self.stats = collections.defaultdict(dict)
		self.model_ema = None
		self.ema_rate = None
		self.n_iters = None
		self.pbar = None

		self.writer = None
		self.logger = None
		self.dl_trn = None
		self.dl_vld = None
		self.dl_tst = None
		self.setup_data()
		self.data_dependent_init()

		self.optim = None
		self.optim_schedule = None
		self.setup_optim()

		if self.verbose:
			tot = sum([
				p.nelement() for p
				in self.parameters()
			])
			if tot // 1e6 > 0:
				tot = f"{tot / 1e6:0.1f} M"
			elif tot // 1e3 > 0:
				tot = f"{tot / 1e3:0.1f} K"
			print(f"\n# params: {tot}")

	def print(self):
		names_str = '\n'.join([
			str(self.model.cfg.name()),
			f"{self.cfg.name()}_({self.model.timestamp})"
		])
		print(names_str)
		return

	def train(
			self,
			fit_name: str = None,
			epochs: Union[int, range] = None,
			fresh_fit: bool = True,
			save: bool = True, ):
		epochs = epochs if epochs else self.cfg.epochs
		assert isinstance(epochs, (int, range)), "allowed: {int, range}"
		epochs = range(epochs) if isinstance(epochs, int) else epochs
		fit_name = fit_name if fit_name else self.cfg.name()
		if fresh_fit:
			self.stats.clear()
		if save:
			if fresh_fit or self.model.chkpt_dir is None:
				self.model.create_chkpt_dir(fit_name)
			if fresh_fit:
				self.cfg.save(self.model.chkpt_dir)
			writer = str(pjoin(
				self.model.cfg.runs_dir,
				os.path.basename(self.model.chkpt_dir),
			))
			self.writer = SummaryWriter(writer)
			self.logger = make_logger(
				name=type(self).__name__,
				path=self.model.chkpt_dir,
				level=logging.WARNING,
			)
		if self.cfg.scheduler_type == 'cosine' and fresh_fit:
			self.optim_schedule.T_max *= len(self.dl_trn)

		self.pbar = tqdm(
			iterable=epochs,
			dynamic_ncols=True,
			leave=True,
			position=0,
		)
		for epoch in self.pbar:
			avg_loss = self.iteration(epoch)
			msg = ', '.join([
				f"epoch # {epoch + 1:d}",
				f"avg loss: {avg_loss:0.3f}",
			])
			self.pbar.set_description(msg)
			if not save:
				continue
			if (epoch + 1) % self.cfg.chkpt_freq == 0:
				self.save(
					checkpoint=epoch + 1,
					path=self.model.chkpt_dir,
				)
			if (epoch + 1) % self.cfg.eval_freq == 0:
				gstep = (epoch + 1) * len(self.dl_trn)
				self.validate(gstep)
		if self.writer is not None:
			self.writer.close()
		return

	def iteration(self, epoch: int = 0):
		raise NotImplementedError

	@torch.inference_mode()
	def validate(self, epoch: int = None):
		raise NotImplementedError

	def setup_data(self):
		raise NotImplementedError

	def data_dependent_init(self):
		pass

	def swap_model(self, new_model, full: bool = False):
		self.model = new_model.to(self.device).eval()
		if full:
			self.setup_data()
			self.setup_optim()
		return

	def select_model(self, ema: bool = False):
		if ema:
			assert self.model_ema is not None
			return self.model_ema.eval()
		return self.model.eval()

	def reset_model(self):
		raise NotImplementedError

	def update_ema(self):
		if self.model_ema is None:
			return
		looper = zip(
			self.model.parameters(),
			self.model_ema.parameters(),
		)
		for p1, p2 in looper:
			p2.data.mul_(self.ema_rate)
			p2.data.add_(p1.data.mul(1-self.ema_rate))
		return

	def parameters(self, requires_grad: bool = True):
		if requires_grad:
			return filter(
				lambda p: p.requires_grad,
				self.model.parameters(),
			)
		else:
			return self.model.parameters()

	def save(self, path: str, checkpoint: int = None):
		if checkpoint is not None:
			global_step = checkpoint * len(self.dl_trn)
		else:
			global_step = None
		state_dict = {
			'metadata': {
				'timestamp': self.model.timestamp,
				'checkpoint': checkpoint,
				'global_step': global_step,
				'stats_trainer': dict(self.stats),
				'stats_model': dict(self.model.stats),
				'root': path},
			'model': self.model.state_dict(),
			'model_ema': self.model_ema.state_dict()
			if self.model_ema is not None else None,
			'optim': self.optim.state_dict(),
			'scheduler': self.optim_schedule.state_dict()
			if self.optim_schedule is not None else None,
		}
		fname = '+'.join([
			type(self.model).__name__,
			type(self).__name__],
		)
		if checkpoint is not None:
			fname = '-'.join([
				fname,
				f"{checkpoint:04d}"
			])
		fname = f"{fname}_({now(True)}).pt"
		fname = pjoin(path, fname)
		torch.save(state_dict, fname)
		return

	def setup_optim(self):
		# optimzer
		kws = dict(
			params=self.parameters(),
			lr=self.cfg.lr,
			**self.cfg.optimizer_kws,
		)
		if self.cfg.optimizer == 'adamax':
			self.optim = torch.optim.Adamax(**kws)
		elif self.cfg.optimizer == 'adamax_fast':
			from .optim import Adamax
			self.optim = Adamax(**kws)
		elif self.cfg.optimizer == 'adam':
			self.optim = torch.optim.Adam(**kws)
		elif self.cfg.optimizer == 'adamw':
			self.optim = torch.optim.AdamW(**kws)
		elif self.cfg.optimizer == 'radam':
			self.optim = torch.optim.RAdam(**kws)
		else:
			raise NotImplementedError(self.cfg.optimizer)

		# scheduler
		if self.cfg.scheduler_type == 'cosine':
			self.optim_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
				self.optim, **self.cfg.scheduler_kws)
		elif self.cfg.scheduler_type == 'exponential':
			self.optim_schedule = torch.optim.lr_scheduler.ExponentialLR(
				self.optim, **self.cfg.scheduler_kws)
		elif self.cfg.scheduler_type == 'step':
			self.optim_schedule = torch.optim.lr_scheduler.StepLR(
				self.optim, **self.cfg.scheduler_kws)
		elif self.cfg.scheduler_type == 'cyclic':
			self.optim = torch.optim.SGD(
				params=self.parameters(),
				lr=self.cfg.lr,
				momentum=0.9,
				weight_decay=self.cfg.optimizer_kws.get('weight_decay', 0),
			)
			self.optim_schedule = torch.optim.lr_scheduler.CyclicLR(
				self.optim, **self.cfg.scheduler_kws)
		elif self.cfg.scheduler_type is None:
			self.optim_schedule = None
		else:
			raise NotImplementedError(self.cfg.scheduler_type)
		return

	def info(self, filter_dirs: bool = True):
		txt = '\n'.join([
			'_' * 100,
			'[INFO] Checkpoint Directory:',
			f"{self.model.chkpt_dir}\n\n",
		])
		# model
		cfg_model = vars(self.model.cfg)
		if filter_dirs:
			cfg_model = {
				key: v for key, v
				in cfg_model.items()
				if '_dir' not in key
			}
		cfg_model = _remove_ranges(cfg_model)
		txt += f"{'_' * 100}\n[INFO] model name: {self.model.cfg.name()}"
		txt += f"\n[INFO] model cfg:\n{json.dumps(cfg_model, indent=4)}"
		# trainer
		txt += f"\n\n{'_' * 100}\n[INFO] trainer name: {self.cfg.name()}"
		txt += f"\n[INFO] trainer cfg:\n{json.dumps(vars(self.cfg), indent=4)}"
		return txt

	def to(self, x, dtype=torch.float32) -> Union[torch.Tensor, List[torch.Tensor]]:
		kws = dict(device=self.device, dtype=dtype)
		if isinstance(x, (tuple, list)):
			return [
				e.to(**kws) if torch.is_tensor(e)
				else torch.tensor(e, **kws)
				for e in x
			]
		else:
			if torch.is_tensor(x):
				return x.to(**kws)
			else:
				return torch.tensor(x, **kws)


def check_grads(
		grads: List[float],
		thres: float,
		gstep: int,
		fn: Callable = np.max, ):
	if fn(grads) > thres:
		msg = 'diverging grad encountered:\n'
		msg += f'fn: {fn.__name__}, grad: {fn(grads):0.1f} > {thres}'
		msg += f'\nglobal step = {gstep}, skipping . . . '
		print(msg)
		return True


def check_nans(loss, gstep: int, verbose: bool = True):
	if torch.isnan(loss).sum().item():
		msg = 'nan encountered in loss. '
		msg += 'optimizer will detect this & skip. '
		msg += f"global step = {gstep}"
		if verbose:
			print(msg)
		return True
	else:
		return False


def _remove_ranges(obj):
	if isinstance(obj, dict):
		return {
			k: _remove_ranges(v) for k, v in obj.items()
			if not isinstance(v, range)}
	elif isinstance(obj, list) or isinstance(obj, tuple):
		return [
			_remove_ranges(item) for item in obj
			if not isinstance(item, range)]
	else:
		return obj
