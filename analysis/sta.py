from utils.generic import *
from main.train import Trainer
from main.datafeeder import FEEDER_CLASSES


def perform_sta_analysis(
		tr: Trainer,
		n_lags: int = 80,
		n_trials: int = 100,
		t_warmup: int = 40,
		t_total: int = 1,
		**kwargs, ):
	feeder_defaults = dict(
		size=tr.model.cfg.input_sz,
		#
		n_repeats=1,
		batch_size=10_000,
		noise_sigma=1.0,
		#
		device=tr.device,
		flatten=True,
	)
	kwargs = setup_kwargs(feeder_defaults, kwargs)
	assert kwargs['n_repeats'] == 1, "for now"

	sta_unnormalized, samples_sum = [], []
	total_timepoints = t_warmup + n_lags + t_total
	shape = (kwargs['batch_size'], total_timepoints)

	for i in tqdm(range(n_trials)):
		tr.model.cfg.set_seeds(i)
		feeder = FEEDER_CLASSES['white-noise'](seed=i, **kwargs)
		tr.model.reset_inference_state(len(feeder))

		samples = torch.zeros(
			(*shape, tr.model.cfg.n_latents[0]),
			dtype=torch.float,
			device=tr.device,
		)
		stim = torch.zeros(
			(*shape, np.prod(tr.model.cfg.input_sz)),
			dtype=torch.float,
			device=tr.device,
		)

		for t in range(total_timepoints):
			tr.model.update_time(t)
			stim[:, t] = feeder[t]
			output = tr.model.xtract_ftr(
				stim[:, t], return_extras=['samples'])
			samples[:, t] = output['samples']

		samples = samples.flatten(end_dim=1)
		stim = stim.flatten(end_dim=1)

		good_inds = []
		for b in range(kwargs['batch_size']):
			good_inds_batch = torch.arange(
				t_warmup + n_lags + kwargs['n_repeats'],
				total_timepoints + 1,
				kwargs['n_repeats'],
			) - 1
			good_inds_batch += b * total_timepoints
			good_inds.append(good_inds_batch)
		good_inds = tr.to(torch.cat(good_inds), dtype=torch.int)

		sta_unnormalized.append(compute_sta(
			n_lags=n_lags,
			stim=stim,
			spks=samples,
			inds=good_inds,
			normalize=False,
			verbose=False,
		))
		sam_sum = samples[good_inds].sum(0)
		sam_sum = sam_sum.reshape((-1, 1, 1))
		samples_sum.append(sam_sum)

	sam_sum = sum(samples_sum)
	sta = sum(sta_unnormalized) / sam_sum
	return sta


def compute_sta(
		n_lags: int,
		stim: torch.Tensor,
		spks: torch.Tensor,
		inds: torch.Tensor = None,
		normalize: bool = True,
		nanzero: bool = True,
		verbose: bool = False, ):
	assert n_lags >= 0
	shape = stim.shape
	nc = spks.shape[-1]
	sta_shape = (nc, n_lags + 1, *shape[1:])
	shape = (nc,) + (1,) * len(shape)

	if inds is None:
		inds = torch.arange(len(stim))
	else:
		inds = inds.clone()
	inds = inds[inds > n_lags]
	sta = torch.zeros(sta_shape, device=stim.device)

	for t in tqdm(inds, disable=not verbose):
		# zero n_lags allowed:
		x = stim[t - n_lags: t + 1]
		y = spks[t]
		mask = y > 0
		if mask.any():
			# TODO: x spatial dim, make it general
			#  so it works for any stim shape
			sta[mask] += torch.einsum(
				'n, tx -> ntx', y[mask], x)

	if normalize:
		n = spks[inds].sum(0)
		n = n.reshape(shape)
		sta /= n
	if nanzero:
		nans = torch.isnan(sta)
		sta[nans] = 0.0
		if nans.sum() and verbose:
			warnings.warn(
				"NaN in STA",
				RuntimeWarning,
			)
	return sta
