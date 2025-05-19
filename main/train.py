from base.train_base import *
from base.dataset import make_dataset
from .datafeeder import FEEDER_CLASSES
from .model import MODEL_CLASSES, IPVAE, IGVAE
from analysis.sparse import sparse_score
from figures.imgs import make_grid


class _BaseTrainer(BaseTrainer):
	def __init__(
			self,
			model: IPVAE,
			cfg: ConfigTrain,
			**kwargs,
	):
		super(_BaseTrainer, self).__init__(
			model=model, cfg=cfg, **kwargs)
		self.n_iters = (
			self.cfg.epochs *
			len(self.dl_trn)
		)
		# kl balancer
		if self.cfg.kl_balancer is not None:
			alphas = kl_balancer_coeff(
				t=self.model.layer.t_total,
				fun=self.cfg.kl_balancer,
				normalize=False,
				flip=True,
			)
			self.alphas = self.to(alphas)
		else:
			self.alphas = None
		# kl time adjuster
		if self.cfg.kl_time_adjuster is not None:
			alphas_time = kl_balancer_coeff(
				t=self.model.layer.t_total,
				fun=self.cfg.kl_time_adjuster,
				normalize=True,
				flip=False,
			)
			self.alphas_time = self.to(alphas_time)
		else:
			self.alphas_time = None
		# kl anneal
		if self.cfg.kl_anneal_cycles == 0:
			self.betas = anneal_linear(
				n_iters=self.n_iters,
				final_value=self.model.cfg.beta_outer,
				anneal_portion=self.cfg.kl_anneal_portion,
				constant_portion=self.cfg.kl_const_portion,
				min_value=self.cfg.kl_beta_min,
			)
		else:
			betas = anneal_cosine(
				n_iters=self.n_iters,
				n_cycles=self.cfg.kl_anneal_cycles,
				portion=self.cfg.kl_anneal_portion,
				start=np.arccos(
					1 - 2 * self.cfg.kl_beta_min
					/ self.model.cfg.beta_outer) / np.pi,
				final_value=self.model.cfg.beta_outer,
			)
			beta_cte = int(np.round(
				self.cfg.kl_const_portion * self.n_iters))
			beta_cte = np.ones(beta_cte) * self.cfg.kl_beta_min
			self.betas = np.insert(betas, 0, beta_cte)[:self.n_iters]
		self.epsilons = anneal_linear(
			min_value=0.0,
			final_value=1.0,
			n_iters=self.n_iters,
			anneal_portion=self.cfg.kl_anneal_portion,
			constant_portion=self.cfg.kl_const_portion,
		)
		if self.cfg.temp_anneal_portion > 0.0:
			kws = dict(
				n_iters=self.n_iters,
				portion=self.cfg.temp_anneal_portion,
				t0=self.cfg.temp_start,
				t1=self.cfg.temp_stop,
			)
			if self.cfg.temp_anneal_type == 'lin':
				self.temperatures = temp_anneal_linear(**kws)
			elif self.cfg.temp_anneal_type == 'exp':
				self.temperatures = temp_anneal_exp(**kws)
			else:
				raise ValueError(self.cfg.temp_anneal_type)
		else:
			self.temperatures = np.ones(self.n_iters)
			self.temperatures *= self.cfg.temp_stop
		# for logging purposes
		self.time_keys = {
			'first': str(0),
			'last': str(self.model.layer.t_total - 1),
			'avg': 'avg',
		}

	def iteration(self, epoch: int = 0):
		raise NotImplementedError

	def _clip_grad(self, is_warming_up: bool):
		if self.cfg.grad_clip is not None:
			if is_warming_up:
				max_norm = self.cfg.grad_clip * 3
			else:
				max_norm = self.cfg.grad_clip
			grad_norm = nn.utils.clip_grad_norm_(
				parameters=self.parameters(),
				max_norm=max_norm,
			).item()
		else:
			grad_norm = np.nan
		return grad_norm

	def _write(self, step, nelbo, grads, r_max, n_exp):
		cond_write = (
			step > 0 and
			self.writer is not None and
			step % self.cfg.log_freq == 0
		)
		if not cond_write:
			return
		to_write = {
			'coeffs/beta': self.betas[step],
			'coeffs/temp': self.temperatures[step],
			'coeffs/lr': self.optim.param_groups[0]['lr'],
		}
		if self.model.cfg.type == 'poisson':
			for t, k in self.time_keys.items():
				to_write.update({
					f'coeffs/r_max/T={t}': r_max[k].avg,
					f'coeffs/n_exp/T={t}': n_exp[k].val,
				})
		for t, k in self.time_keys.items():
			to_write[f'train/nelbo/T={t}'] = nelbo[k].avg
		if self.cfg.grad_clip is not None:
			to_write['train/grad_norm'] = grads.avg
		# write
		for k, v in to_write.items():
			self.writer.add_scalar(k, v, step)
		return

	@torch.no_grad()
	def validate(self, epoch: int = None):
		raise NotImplementedError

	@torch.no_grad()
	def sample(self, n_samples: int, temp: float):
		raise NotImplementedError

	def setup_data(self, gpu: bool = True):
		# create datasets
		device = self.device if gpu else None
		if self.model.cfg.dataset_name in ['ImageNet32', 'CelebA']:
			device = None  # too large to put on gpu
		kws = dict(
			dataset=self.model.cfg.dataset_name,
			load_dir=self.model.cfg.data_dir,
			device=device,
		)
		if self.cfg.dataset_kws is not None:
			kws = {**kws, **self.cfg.dataset_kws}
		trn, vld, tst = make_dataset(**kws)
		# create dataloaders
		kws = dict(
			batch_size=self.cfg.batch_size,
			drop_last=self.shuffle,
			shuffle=self.shuffle,
		)
		self.dl_trn = torch.utils.data.DataLoader(trn, **kws)
		kws.update({'drop_last': False, 'shuffle': False})
		self.dl_vld = torch.utils.data.DataLoader(vld, **kws)
		self.dl_tst = torch.utils.data.DataLoader(tst, **kws)
		return

	def get_feeder(self, x: torch.Tensor = None, **kwargs):
		# (1) setup default args
		defaults = self.model.cfg.feeder_cfg
		kwargs = setup_kwargs(defaults, kwargs)
		# (2) get some data batch if not provided
		if x is None:
			x = next(iter(self.dl_vld))[0]
		# (3) send to device
		if x.device != self.device:
			x = self.to(x)
		# (4) construct feeder object
		feeder = self.model.cfg.feeder
		feeder = FEEDER_CLASSES[feeder]
		feeder = feeder(x, **kwargs)
		return feeder

	@torch.no_grad()
	def show_recon(
			self,
			t: float = None,
			n_samples: int = 16,
			display: bool = True,
			**kwargs, ):
		defaults = dict(
			dpi=100,
			figsize=(8, 1),
			cmap='Greys_r',
			normalize=False,
			pad=0,
		)
		kwargs = setup_kwargs(defaults, kwargs)

		x = next(iter(self.dl_vld))[0]
		x = self.to(x)[:n_samples]

		if t is not None:
			t_original = self.model.layer.temp.item()
			self.model.update_temp(t)
		else:
			t_original = None

		# get recon
		y = self.model(x)[-1]

		if t is not None:
			self.model.update_temp(t_original)

		g2p = make_grid(
			torch.cat([x, y]),
			grid_size=(2, n_samples),
			pad=kwargs['pad'],
			normalize=kwargs['normalize'],
		)

		fig, ax = create_figure(
			figsize=kwargs['figsize'],
			dpi=kwargs['dpi'],
		)
		ax.imshow(g2p, cmap=kwargs['cmap'])
		remove_ticks(ax)
		if display:
			plt.show()
		else:
			plt.close()
		return fig, ax

	@torch.no_grad()
	def show_samples(
			self,
			t: float = None,
			nrows: int = 10,
			display: bool = True,
			**kwargs, ):
		defaults = dict(
			dpi=100,
			figsize=(3.5, 3.5),
			cmap='Greys_r',
			normalize=False,
			pad=0,
		)
		kwargs = setup_kwargs(defaults, kwargs)

		# sample
		x, _ = self.sample(nrows ** 2, t)
		g2p = make_grid(
			x=x[:nrows ** 2],
			grid_size=nrows,
			pad=kwargs['pad'],
			normalize=kwargs['normalize'],
		)
		fig, ax = create_figure(
			figsize=kwargs['figsize'],
			dpi=kwargs['dpi'],
		)
		ax.imshow(g2p, cmap=kwargs['cmap'])
		remove_ticks(ax)
		if display:
			plt.show()
		else:
			plt.close()
		return fig, ax

	def show_schedules(self):
		fig, axes = create_figure(
			nrows=1, ncols=3,
			figsize=(6.5, 1.6),
			layout='constrained',
		)
		# temp
		lbl = f"t0 = {self.cfg.temp_start:0.2g}"
		axes[0].plot(self.temperatures, color='k', lw=3)
		axes[0].axhline(
			self.cfg.temp_start,
			label=lbl,
			color='g',
			ls='--',
		)
		lbl = f"t1 = {self.cfg.temp_stop:0.2g}"
		axes[0].axhline(
			self.cfg.temp_stop,
			label=lbl,
			color='r',
			ls='--',
		)
		# beta
		lbl = r"$\beta = $" + f"{self.model.cfg.beta_outer:0.4g}"
		axes[1].plot(self.betas, color='C0', lw=3)
		axes[1].axhline(
			self.model.cfg.beta_outer,
			label=lbl,
			color='g',
			ls='--',
		)
		# lateral epsilon
		lbl = r"$\epsilon = $" + f"{self.epsilons[-1]:0.2g}"
		axes[2].plot(self.epsilons, color='C4', lw=3)
		axes[2].axhline(
			self.epsilons[-1],
			label=lbl,
			color='g',
			ls='--',
		)

		add_legend(axes, fontsize=10)
		for ax in axes.flat:
			ax.ticklabel_format(
				axis='x',
				style='sci',
				scilimits=(0, 0),
			)
		plt.show()
		return

	def reset_model(self):
		raise NotImplementedError


class Trainer(_BaseTrainer):
	def __init__(
			self,
			model: Union[
				IPVAE, IGVAE],
			cfg: ConfigTrain,
			**kwargs,
	):
		super(Trainer, self).__init__(
			model=model, cfg=cfg, **kwargs)
		self._init_fun()

		if self.cfg.ema_rate is not None:
			model_class = MODEL_CLASSES[self.model.cfg.type]
			model_ema = model_class(self.model.cfg)
			self.model_ema = model_ema.to(self.device).eval()
			self.ema_rate = self.to(self.cfg.ema_rate)

	def iteration(self, epoch: int = 0):
		self.model.train()
		timer = Timer()

		seq = range(self.model.layer.t_total)
		keys = [str(t) for t in seq] + ['avg']
		r_max = AverageMeterDict(keys)
		n_exp = AverageMeterDict(keys)
		nelbo = AverageMeterDict(keys)
		grads = AverageMeter()

		timer("starting loop")
		for i, (x, *_) in enumerate(self.dl_trn):
			timer("start of batch")
			gstep = epoch * len(self.dl_trn) + i

			# feeder
			if x.device != self.device:
				x = self.to(x)
			feeder = self.get_feeder(x, seed=gstep)

			# warmup lr?
			progress = gstep / self.n_iters
			is_warming_up = progress < self.cfg.warmup_portion
			if is_warming_up:
				lr = (
					self.cfg.lr * progress /
					self.cfg.warmup_portion
				)
				for param_group in self.optim.param_groups:
					param_group['lr'] = lr

			# set model params (temperature, eps, etc...)
			self.model.update_temp(self.temperatures[gstep])
			self.model.update_eps(self.epsilons[gstep])

			loss_np = np_nans(seq.stop)
			r_max_np = np_nans(seq.stop)
			n_exp_np = np_nans(seq.stop)

			# zero grad
			self.optim.zero_grad()
			# local loop begins
			loss_seq, counter = 0, 0
			for t in seq:
				self.model.update_time(t)
				lstep = gstep * seq.stop + t
				# forward pass
				loss, posterior = self._fun(
					beta=self.betas[gstep],
					inpt=feeder[t],
				)
				loss_seq += loss
				counter += 1
				timer(f"forward+loss_t={t}")

				# collect stats
				with torch.no_grad():
					self.stats['loss'][lstep] = loss.item()
					loss_np[t] = loss.item()
					if self.model.cfg.type == 'poisson':
						r_max_np[t] = posterior.rate.max().item()
						n_exp_np[t] = posterior.n_exp

				# backward?
				if (t + 1) == self.model.cfg.t_train:
					loss = loss_seq / counter
					loss.backward()
					# grad clip
					grad_norm = self._clip_grad(is_warming_up)
					# optim step
					self.optim.step()
					self.optim.zero_grad()
					# important: detach state
					self.model.detach_inference_state()
					# collect stats
					with torch.no_grad():
						self.stats['grad'][lstep] = grad_norm
						grads.update(grad_norm)
					# reset buffer
					loss_seq, counter = 0, 0
					timer(f"backward_t={t}")

			# optim schedule step
			cond_schedule = (
				self.optim_schedule is not None
				and not is_warming_up
			)
			if cond_schedule:
				self.optim_schedule.step()

			# update average meters & stats
			with torch.no_grad():
				nelbo.update(_prep(loss_np))
				if self.model.cfg.type == 'poisson':
					r_max.update(_prep(r_max_np))
					n_exp.update(_prep(n_exp_np))

			# n_exp gets updated once per ~200 iters
			cond_update_n_exp = (
				gstep % (self.cfg.log_freq * 20) == 0
				and self.model.cfg.type == 'poisson'
				and not is_warming_up
			)
			if cond_update_n_exp:
				max_rates = [
					r_max[str(t)].avg
					for t in seq
				]
				self.model.layer.update_n_exp(
					max_rates)

			# save more stats
			self.stats['lr'][gstep] = \
				self.optim.param_groups[0]['lr']
			self.stats['t'][gstep] = self.model.temp.item()
			if self.model.cfg.type == 'poisson':
				self.stats['r_max'][gstep] = r_max['avg'].avg
				self.stats['n_exp'][gstep] = n_exp['avg'].val
			# write?
			self._write(gstep, nelbo, grads, r_max, n_exp)
			# reset avg meters once per ~200 iters
			if gstep % (self.cfg.log_freq * 20) == 0:
				grads.reset()
				nelbo.reset()

		# print timer results?
		if self.verbose:
			timer.print()
		return nelbo['avg'].avg

	@torch.no_grad()
	def validate(
			self,
			gstep: int = None,
			n_samples: int = 4096,
			**kwargs, ):
		# forward
		loss, samples = self.forward('vld', **kwargs)

		if gstep is not None:
			# kl perdim
			div = sum(self.model.cfg.n_latents)
			kl_perdim = _prep(loss['kl'].mean(0), div)
			# recon perdim
			div = np.prod(self.model.cfg.input_sz)
			recon_perdim = _prep(loss['recon'].mean(0), div)
			# loss perdim
			loss_perdim = {
				'kl': kl_perdim,
				'recon': recon_perdim,
				'nelbo': {
					k: v + kl_perdim[k] for
					k, v in recon_perdim.items()
				},
			}
			to_write = {}
			for name, val_dict in loss_perdim.items():
				for t, k in self.time_keys.items():
					to_write[f"eval/{name}/T={t}"] = val_dict[k]

			# sparsity analysis
			_samples = {
				str(t): samples[:, t, :] for t in
				range(self.model.layer.t_total)
			}
			_samples['avg'] = flatten_np(samples, end_dim=1)
			for t, k in self.time_keys.items():
				sprs_results = sparse_score(_samples[k], cutoff=0.01)
				lifetime, population, percents = sprs_results
				to_write.update({
					f"sprs/%-zero/T={t}": percents.get('0', np.nan),
					f"sprs/lifetime/T={t}": np.nanmean(lifetime),
					f"sprs/population/T={t}": np.nanmean(population),
				})

			# write stuff
			for k, v in sorted(to_write.items()):
				self.writer.add_scalar(k, v, gstep)
				self.stats[k][gstep] = v

			# add figs
			self.figs_to_writer(gstep, kwargs.get('temp'))

		return loss, samples

	@torch.no_grad()
	def forward(
			self,
			dl_name: str,
			use_ema: bool = False,
			verbose: bool = False,
			**kwargs, ):
		assert dl_name in ['trn', 'vld', 'tst']
		dl = getattr(self, f"dl_{dl_name}")
		if dl is None:
			return None

		model = self.select_model(use_ema)
		kwargs.setdefault('return_extras', ['samples'])
		kwargs.setdefault('temp', model.cfg.validation_temp)

		n = len(dl.dataset)
		t = self.model.layer.t_total
		k = self.model.cfg.n_latents[0]
		loss_recon = np.empty((n, t), dtype=float)
		loss_kl = np.empty((n, t, k), dtype=float)
		samples = np.empty((n, t, k), dtype=float)

		# loop over data
		looper = tqdm(
			enumerate(iter(dl)),
			disable=not verbose,
			total=len(dl),
			ncols=70,
		)
		for b, (x, *_) in looper:
			if x.device != self.device:
				x = self.to(x)
			feeder = self.get_feeder(
				x=x, seed=self.model.cfg.seed)
			# which data sampels?
			start = dl.batch_size * b
			samples_intvl = range(start, start + len(x))
			for t in range(self.model.layer.t_total):
				self.model.update_time(t)
				output = model.xtract_ftr(
					feeder[t], **kwargs)
				loss_recon[samples_intvl, t] = tonp(
					output['loss_recon'])
				loss_kl[samples_intvl, t] = tonp(
					output['loss_kl'])
				samples[samples_intvl, t] = tonp(
					output['samples'])
		loss = {
			'recon': loss_recon,
			'kl': loss_kl.mean(-1),
			'kl_diag': loss_kl.mean(0),
		}
		loss['nelbo'] = loss['recon'] + loss['kl']
		return loss, samples

	# noinspection PyTypeChecker
	@torch.no_grad()
	def analysis(
			self,
			dl: Any = 'vld',
			t_total: int = None,
			data_batch_sz: int = None,
			n_data_batches: int = None,
			active: torch.Tensor = None,
			return_recon: bool = False,
			compute_sprs: bool = False,
			avg_samples: bool = True,
			verbose: bool = True,
			**kwargs, ):
		if isinstance(dl, str):
			assert dl in ['trn', 'vld', 'tst']
			dl = getattr(self, f"dl_{dl}", None)
		if dl is None:
			return None
		# create new dataloader w/ shuffle = False
		kws_dl = dict(
			batch_size=data_batch_sz
			or self.cfg.batch_size,
			drop_last=False,
			shuffle=False,
		)
		if isinstance(dl, torch.utils.data.DataLoader):
			dl = dl.dataset
		dl = torch.utils.data.DataLoader(dl, **kws_dl)

		if n_data_batches in [None, -1]:
			n_data_batches = len(dl)

		model = self.select_model(False)
		t_total = t_total or model.layer.t_total

		temp = model.cfg.validation_temp
		kwargs.setdefault('temp', temp)
		extra_items = ['samples', 'du', 'r2', 'mse']
		if return_recon:
			extra_items.append('recon')
		kwargs.setdefault('return_extras', extra_items)

		if active is None:
			active = torch.ones(model.layer.latent_dim)
		active = self.to(active, dtype=torch.bool)

		# create empty arrays to save results
		n = n_data_batches * dl.batch_size
		n = min(n, len(dl.dataset))
		# kl, mse, du_norm
		if avg_samples:
			kl = np.zeros(t_total, dtype=np.float32)
			r2 = np.zeros(t_total, dtype=np.float32)
			mse = np.zeros(t_total, dtype=np.float32)
			du_norm = np.zeros(t_total, dtype=np.float32)
		else:
			shape = (n, t_total)
			kl = np.empty(shape, dtype=np.float32)
			r2 = np.empty(shape, dtype=np.float32)
			mse = np.empty(shape, dtype=np.float32)
			du_norm = np.empty(shape, dtype=np.float32)
		# samples
		k = model.layer.latent_dim
		shape = (n, t_total, k)
		samples = np.empty(shape, dtype=np.float32)
		# final state
		state_final = np.empty((n, k), dtype=np.float32)
		# recon
		if return_recon:
			m = model.layer.input_dim
			shape = (n, t_total, m)
			recon = np.empty(shape, dtype=np.float32)
		else:
			recon = None

		# loop over data
		looper = tqdm(
			enumerate(iter(dl)),
			total=n_data_batches,
			disable=not verbose,
			ncols=70,
		)
		for b, (x, *_) in looper:
			if b >= n_data_batches:
				break
			if x.device != self.device:
				x = self.to(x)
			feeder = self.get_feeder(
				x=x, seed=self.model.cfg.seed)

			# which data samples?
			start = dl.batch_size * b
			samples_intvl = range(start, start + len(x))

			_y, _z, _norms = [], [], []
			_kl_list, _mse_list, _r2_list = [], [], []
			# loop over time
			for t in range(t_total):
				# (1) set model time
				self.model.update_time(t)
				# (2) fwd pass
				output = model.xtract_ftr(feeder[t], **kwargs)
				# (3) save
				if (t + 1) == t_total:
					state_final[samples_intvl] = tonp(
						model.layer.u[0])
				# tonp + append
				_norms_batch = torch.linalg.norm(
					output['du'][:, active], dim=-1)
				if avg_samples:
					_kl_list.append(tonp(torch.sum(torch.sum(
						output['loss_kl'][:, active], dim=-1), dim=0)
					))
					_mse_list.append(tonp(torch.sum(
						output['mse'], dim=0)
					))
					_r2_list.append(tonp(torch.sum(
						output['r2'], dim=0)
					))
					_norms.append(tonp(torch.sum(
						_norms_batch, dim=0)
					))
				else:
					_kl_list.append(tonp(torch.sum(
						output['loss_kl'][:, active], dim=-1)
					))
					_mse_list.append(tonp(output['mse']))
					_r2_list.append(tonp(output['r2']))
					_norms.append(tonp(_norms_batch))
				_z.append(tonp(output['samples']))
				if return_recon:
					_y.append(tonp(output['recon']))
			# stack & append
			_kl_list, _mse_list, _r2_list, _norms = stack_map(
				x=[_kl_list, _mse_list, _r2_list, _norms],
				axis=0 if avg_samples else 1,
			)
			# save: losses + du norm
			if avg_samples:
				kl += _kl_list
				r2 += _r2_list
				mse += _mse_list
				du_norm += _norms
			else:
				kl[samples_intvl] = _kl_list
				r2[samples_intvl] = _r2_list
				mse[samples_intvl] = _mse_list
				du_norm[samples_intvl] = _norms
			# save: samples
			_z = np.stack(_z, axis=1)
			samples[samples_intvl] = _z
			# save: recon
			if return_recon:
				_y = np.stack(_y, axis=1)
				recon[samples_intvl] = _y
		# sum —> mean
		if avg_samples:
			kl /= n
			r2 /= n
			mse /= n
			du_norm /= n
		# (5) sparsity score
		z_active = samples[..., tonp(active)]
		percent_zeros = np.mean(z_active == 0, axis=(0, -1))
		if compute_sprs:
			lifetime, population, _ = sparse_score(
				z=self.to(z_active), cutoff=None)
			lifetime = tonp(lifetime.mean(1))  # mean: neurons
			population = tonp(population.mean(0))  # mean: samples
		else:
			lifetime = np_nans(t_total)
			population = np_nans(t_total)
		# (6) results to return
		results = {
			'kl': kl,
			'r2': r2,
			'mse': mse,
			'nelbo': kl + mse,
			'du_norm': du_norm,
			'lifetime': lifetime,
			'population': population,
			'%-zeros': percent_zeros,
			'state_final': state_final,
			'samples_final': samples[:, -1, :],
		}
		# include recon?
		if return_recon:
			results['recon'] = recon
		return results

	# noinspection PyTypeChecker
	@torch.no_grad()
	def xtract_ftr(
			self,
			dl: Any = 'vld',
			t_total: int = 1000,
			save_freq: int = 100,
			data_batch_sz: int = None,
			n_data_batches: int = None,
			verbose: bool = True,
			**kwargs, ):
		if isinstance(dl, str):
			assert dl in ['trn', 'vld', 'tst']
			dl = getattr(self, f"dl_{dl}", None)
		if dl is None:
			return None
		# create new dataloader w/ shuffle = False
		kws_dl = dict(
			batch_size=data_batch_sz
			or self.cfg.batch_size,
			drop_last=False,
			shuffle=False,
		)
		if isinstance(dl, torch.utils.data.DataLoader):
			dl = dl.dataset
		dl = torch.utils.data.DataLoader(dl, **kws_dl)

		if n_data_batches in [None, -1]:
			n_data_batches = len(dl)

		model = self.select_model(False)
		t_total = t_total or model.layer.t_total

		temp = model.cfg.validation_temp
		kwargs.setdefault('temp', temp)
		extra_items = ['u', 'du', 'samples', 'r2', 'mse']
		kwargs.setdefault('return_extras', extra_items)

		# create empty arrays to save results
		# r2, mse, du_norm
		r2 = np.zeros(t_total, dtype=np.float32)
		mse = np.zeros(t_total, dtype=np.float32)
		du_norm = np.zeros(t_total, dtype=np.float32)
		# state, samples
		assert save_freq >= 1
		save_freq = int(save_freq)
		if save_freq == 1:
			n_save = t_total
			times = np.arange(t_total, dtype=int)
		else:
			n_save = t_total / save_freq
			n_save = int(np.floor(n_save)) + 1
			times = np.zeros(n_save, dtype=int)
		n = n_data_batches * dl.batch_size
		n = min(n, len(dl.dataset))
		k = model.layer.latent_dim
		shape = (n, n_save, k)
		state = np.empty(shape, dtype=np.float32)
		samples = np.empty(shape, dtype=np.float32)

		# loop over data
		looper = tqdm(
			enumerate(iter(dl)),
			total=n_data_batches,
			disable=not verbose,
			ncols=70,
		)
		for b, (x, *_) in looper:
			if b >= n_data_batches:
				break
			if x.device != self.device:
				x = self.to(x)
			feeder = self.get_feeder(
				x=x, seed=self.model.cfg.seed)

			# which data samples?
			start = dl.batch_size * b
			samples_intvl = range(start, start + len(x))

			_u, _z, _norms = [], [], []
			_mse_list, _r2_list = [], []
			# loop over time
			for t in range(t_total):
				# (1) set model time
				self.model.update_time(t)
				# (2) fwd pass
				output = model.xtract_ftr(
					feeder[t], **kwargs)
				# (3) save
				# tonp + append
				_norms_batch = torch.linalg.norm(
					output['du'], dim=-1)
				_norms.append(tonp(torch.sum(
					_norms_batch, dim=0)
				))
				_r2_list.append(tonp(torch.sum(
					output['r2'], dim=0)
				))
				_mse_list.append(tonp(torch.sum(
					output['mse'], dim=0)
				))
				if t == 0 or (t + 1) % save_freq == 0:
					_u.append(tonp(output['u']))
					_z.append(tonp(output['samples']))
					if save_freq > 1:
						times[t // save_freq + 1] = t
			# cat & append
			_r2_list, _mse_list, _norms = stack_map(
				x=[_r2_list, _mse_list, _norms], axis=0)
			# save: losses + du norm
			r2 += _r2_list
			mse += _mse_list
			du_norm += _norms
			# save: samples
			_u, _z = stack_map(
				x=[_u, _z], axis=1)
			state[samples_intvl] = _u
			samples[samples_intvl] = _z
		# sum —> mean
		r2 /= n
		mse /= n
		du_norm /= n
		# (6) results to return
		results = {
			'r2': r2,
			'mse': mse,
			'state': state,
			'samples': samples,
			'du_norm': du_norm,
			'times': times + 1,
		}
		return results

	@torch.no_grad()
	def figs_to_writer(
			self,
			gstep: int,
			order: Sequence[int] = None, ):
		freq = max(10, self.cfg.eval_freq * 5)
		ep = int(gstep / len(self.dl_trn))
		cond = ep % freq == 0
		if not cond:
			return None
		figs = {}
		if self.model.cfg.dec_type in ['lin', 'mlp']:
			if order is None:
				order = self.model.layer.u_init[0]
				order = np.argsort(tonp(order.squeeze()))
			# prep
			cond_balls = 'BALLS' in self.model.cfg.dataset
			method = 'abs-max' if cond_balls else 'min-max'
			dpi = 50 if cond_balls else 90
			# plot
			figs['phi'] = self.model.show(
				which='phi',
				order=order,
				method=method,
				add_title=False,
				display=False,
				dpi=dpi,
			)[0]
		for name, f in figs.items():
			self.writer.add_figure(f'figs/{name}', f, gstep)

		return figs

	def reset_model(self):
		mode_class = MODEL_CLASSES[self.model.cfg.type]
		self.model = mode_class(self.model.cfg).to(self.device)
		if self.model_ema is not None:
			self.model_ema = mode_class(self.model.cfg).to(self.device)
		return

	def _init_fun(self):
		def _fun(inpt, beta):
			# ----------------------------------
			# forward
			output = self.model(inpt)
			# ----------------------------------
			# compute loss
			kl_batch = torch.sum(
				output['loss_kl'], dim=-1)
			loss_batch = (  # shape: [B,]
				output['loss_recon'] +
				kl_batch * beta
			)
			# ----------------------------------
			# Info: math says sum, but implementation can be mean
			# because if sum —> lr & grad clip do not generalize
			loss = torch.mean(loss_batch)
			# ----------------------------------
			# scale loss to the range ~256
			# this was useful for BALLS64
			total_dims = np.prod(self.model.cfg.input_sz)
			scale_factor = 256.0 / float(total_dims)
			loss = loss * scale_factor
			# ----------------------------------
			return loss, output['posterior']

		self._fun = _fun
		return


@torch.no_grad()
def _prep(vals, div: float = None):
	"""
	vals: has the same length as seq_len
	"""
	vals = {str(t): v for t, v in enumerate(vals)}
	vals['avg'] = sum(vals.values()) / len(vals)
	if div is not None:
		vals = {
			k: v / float(div) for
			k, v in vals.items()
		}
	return vals


def save_fit_info(
		tr: Trainer,
		args: dict,
		start: str,
		stop: str = None,
		save_dir: str = 'logs',
		root: str = 'Dropbox/git/_IterativeVAE', ):
	stop = stop or now(True)
	# make info string
	host = os.uname().nodename
	done = f"[PROGRESS] fitting VAE on {host}-cuda:{args['device']} done!"
	done += f" run time  ——>  {time_dff_string(start, stop)}  <——\n"
	done += f"[PROGRESS] start: {start}  ———  stop: {stop}\n"
	info = tr.info()
	info += f"\n\n{'_' * 100}\n[INFO] args:\n{json.dumps(args, indent=4)}"
	info = f"{done}\n{info}"

	# file name
	fname = [
		tr.model.cfg.str_model,
		args['archi'] if
		args['archi'].endswith('>')
		else f"<{args['archi']}>",
		tr.model.cfg.dataset,
		tr.model.cfg.str_t,
		tr.model.cfg.str_b,
		tr.model.cfg.str_k,
	]
	if args.get('comment') is not None:
		fname += [args['comment']]
	fname += [
		f"{host}-{tr.model.cfg.seed}",
		f"({tr.model.timestamp})",
	]
	fname = '_'.join(fname)
	# save dir
	save_dir = pjoin(add_home(root), save_dir)
	os.makedirs(save_dir, exist_ok=True)
	# save
	save_obj(
		obj=info,
		file_name=fname,
		save_dir=save_dir,
		mode='txt',
	)
	return


def _setup_args() -> argparse.Namespace:
	from main.config import (
		DATA_CHOICES, T_ANNEAL_CHOICES
	)
	parser = argparse.ArgumentParser()

	#####################
	# setup
	#####################
	parser.add_argument(
		"device",
		choices=range(torch.cuda.device_count()),
		help='cuda:device',
		type=int,
	)
	parser.add_argument(
		"dataset",
		choices=DATA_CHOICES,
		type=str,
	)
	parser.add_argument(
		"model",
		choices=MODEL_CLASSES,
		type=str,
	)
	########################
	# vae cfgs (common)
	########################
	parser.add_argument(
		"--archi",
		help='architecture type',
		default='grad|lin',
		type=str,
	)
	parser.add_argument(
		"--t_train",
		help='# time points',
		default=16,
		type=int,
	)
	parser.add_argument(
		"--n_iters_outer",
		help='outer loop: learning',
		default=1,
		type=int,
	)
	parser.add_argument(
		"--n_iters_inner",
		help='inner loop: inference',
		default=1,
		type=int,
	)
	parser.add_argument(
		"--beta_outer",
		help='beta (outer loop)',
		default=16.0,
		type=float,
	)
	parser.add_argument(
		"--beta_inner",
		help='beta (inner loop)',
		default=1.0,
		type=float,
	)
	parser.add_argument(
		"--n_latents",
		help='# latents',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, int),
	)
	parser.add_argument(
		"--init_dist",
		help='init dist',
		default='normal',
		type=str,
	)
	parser.add_argument(
		"--init_scale",
		help='init scale',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, float),
	)
	parser.add_argument(
		"--clamp_du",
		help='softclamp on du',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, float),
	)
	parser.add_argument(
		"--clamp_u",
		help='softclamp on u',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, float),
	)
	parser.add_argument(
		"--fit_prior",
		help='fit prior?',
		default=True,
		type=true_fn,
	)
	parser.add_argument(
		"--stochastic",
		help='pred = sample conditional?',
		default=True,
		type=true_fn,
	)
	parser.add_argument(
		"--seed",
		help='random seed',
		default=0,
		type=int,
	)
	########################
	# vae cfgs (specific)
	########################
	# poisson
	parser.add_argument(
		"--clamp_prior",
		help='prior init clamp',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, float),
	)
	parser.add_argument(
		"--prior_log_dist",
		help='prior dist type',
		default='uniform',
		type=str,
	)
	# gaussian & laplace
	parser.add_argument(
		"--latent_act",
		help='activation on z?',
		default=None,
		type=str,
	)
	########################
	# trainer cfgs
	########################
	parser.add_argument(
		"--lr",
		help='learning rate',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, float),
	)
	parser.add_argument(
		"--epochs",
		help='# epochs',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, int),
	)
	parser.add_argument(
		"--batch_size",
		help='batch size',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, int),
	)
	parser.add_argument(
		"--warm_restart",
		help='# warm restarts',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, int),
	)
	parser.add_argument(
		"--warmup_portion",
		help='warmup portion',
		default=0.005,
		type=float,
	)
	parser.add_argument(
		"--optimizer",
		help='optimizer',
		default='adamax_fast',
		type=str,
	)
	# temp
	parser.add_argument(
		"--temp_start",
		help='temp: [start] —> stop',
		default=1.0,
		type=float,
	)
	parser.add_argument(
		"--temp_stop",
		help='temp: start —> [stop]',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, float),
	)
	parser.add_argument(
		"--temp_anneal_type",
		choices=T_ANNEAL_CHOICES,
		help='temp anneal type',
		default='lin',
		type=str,
	)
	parser.add_argument(
		"--temp_anneal_portion",
		help='temp anneal portion',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, float),
	)
	# kl
	parser.add_argument(
		"--kl_anneal_portion",
		help='kl beta anneal portion',
		default=0.1,
		type=float,
	)
	parser.add_argument(
		"--kl_const_portion",
		help='kl const portion',
		default=1e-3,
		type=float,
	)
	parser.add_argument(
		"--kl_anneal_cycles",
		help='0: linear, >0: cosine',
		default=0,
		type=int,
	)
	parser.add_argument(
		"--grad_clip",
		help='gradient norm clipping',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, float),
	)
	parser.add_argument(
		"--chkpt_freq",
		help='checkpoint freq',
		default=10,
		type=int,
	)
	parser.add_argument(
		"--eval_freq",
		help='eval freq',
		default=20,
		type=int,
	)
	parser.add_argument(
		"--log_freq",
		help='log freq',
		default=10,
		type=int,
	)
	parser.add_argument(
		"--comment",
		help='comment',
		default=None,
		type=str,
	)
	parser.add_argument(
		"--dry_run",
		help='to make sure config is alright',
		action='store_true',
		default=False,
	)
	parser.add_argument(
		"--dont_save_info",
		help='saves info by default',
		action='store_true',
		default=False,
	)
	parser.add_argument(
		"--cudnn_bench",
		help='use cudnn benchmark?',
		action='store_true',
		default=False,
	)
	parser.add_argument(
		"--verbose",
		help='to make sure config is alright',
		action='store_true',
		default=False,
	)
	return parser.parse_args()


def _main():
	args = _setup_args()
	cfg_vae, cfg_tr = default_configs(
		dataset=args.dataset,
		model_type=args.model,
		archi_type=args.archi,
	)
	# filter: removes unrelated kwargs
	cfg_vae = filter_kwargs(CFG_CLASSES[args.model], cfg_vae)
	cfg_tr = filter_kwargs(ConfigTrain, cfg_tr)
	# convert from dict to cfg objects, back to dict
	# this step adds the full set of default values
	cfg_vae['save'] = False
	cfg_vae = obj_attrs(CFG_CLASSES[args.model](**cfg_vae))
	cfg_tr = obj_attrs(ConfigTrain(**cfg_tr))
	# override with values provided by user
	for k, v in vars(args).items():
		if v == '__placeholder__':
			continue
		if k in cfg_vae:
			cfg_vae[k] = v
		if k in cfg_tr:
			cfg_tr[k] = v
	# final convert: from dict to cfg objects
	cfg_vae['save'] = not args.dry_run
	cfg_vae = CFG_CLASSES[args.model](**cfg_vae)
	cfg_tr = ConfigTrain(**cfg_tr)

	# model & tr
	device = f"cuda:{args.device}"
	model = MODEL_CLASSES[args.model](cfg_vae)
	tr = Trainer(model, cfg_tr, device=device)

	if args.verbose:
		print(args)
		tr.model.print()
		print(f"# train iters: {tr.n_iters:,}\n")
		tr.print()
		print()

	if args.comment is not None:
		comment = '_'.join([
			args.comment,
			tr.cfg.name(),
		])
	else:
		comment = tr.cfg.name()

	if args.cudnn_bench:
		torch.backends.cudnn.benchmark = True
		torch.backends.cudnn.benchmark_limit = 0

	start = now(True)
	if not args.dry_run:
		tr.train(comment)
		if not args.dont_save_info:
			save_fit_info(
				tr=tr,
				args=vars(args),
				start=start,
			)
	return


if __name__ == "__main__":
	_main()
