from utils.generic import *
from main.train import Trainer
from sklearn import linear_model
from sklearn.metrics import (
	r2_score, classification_report)


def clf_score(
		trainer: Trainer,
		t_total: int = 1000,
		save_freq: int = 100,
		data_batch_sz: int = 5000,
		verbose: bool = True, ):
	# process fit, get results
	results, times, ground_truths = _process(
		trainer=trainer,
		t_total=t_total,
		save_freq=save_freq,
		data_batch_sz=data_batch_sz,
		verbose=verbose,
	)

	accu = collections.defaultdict(dict)
	for i, t in times.items():
		u = results['trn']['state'][:, i, :]
		lr = linear_model.LogisticRegression().fit(
			X=_u2z(u, trainer), y=ground_truths['trn'])
		for k, result_dict in results.items():
			u = result_dict['state'][:, i, :]
			accu[k][t] = classification_report(
				y_true=ground_truths[k],
				y_pred=lr.predict(_u2z(u, trainer)),
				output_dict=True,
			)['accuracy']

	output = {
		'clf_accuracy': dict(accu),
		'results_xtract': results,
	}
	return output


def untangle_score(
		trainer: Trainer,
		t_total: int = 1000,
		save_freq: int = 100,
		data_batch_sz: int = 2000,
		verbose: bool = True, ):
	# process fit, get results
	results, times, ground_truths = _process(
		trainer=trainer,
		t_total=t_total,
		save_freq=save_freq,
		data_batch_sz=data_batch_sz,
		verbose=verbose,
	)

	r2 = collections.defaultdict(dict)
	corr = collections.defaultdict(dict)

	for i, t in times.items():
		u = results['trn']['state'][:, i, :]
		lr = linear_model.LinearRegression().fit(
			X=_u2z(u, trainer), y=ground_truths['trn'])
		for k in ['trn', 'vld', 'tst']:
			u = results[k]['state'][:, i, :]
			# get pred & true
			pred = lr.predict(_u2z(u, trainer))
			true = ground_truths[k]
			# compute scores
			r2[k][t] = r2_score(
				y_true=true,
				y_pred=pred,
				multioutput='raw_values',
			)
			corr[k][t] = 1 - np.diag(sp_dist.cdist(
				XA=true.T,
				XB=pred.T,
				metric='correlation',
			))

	output = {
		'r2_zg': dict(r2),
		'corr_zg': dict(corr),
		'results_xtract': results,
	}
	return output


def _process(
		trainer: Trainer,
		t_total: int = 100,
		save_freq: int = 10,
		data_batch_sz: int = 2000,
		verbose: bool = True, ):

	kws = dict(
		t_total=t_total,
		save_freq=save_freq,
		data_batch_sz=data_batch_sz,
		verbose=verbose,
	)

	dataloader_keys = ['trn', 'vld']
	if trainer.dl_tst.dataset is not None:
		dataloader_keys.append('tst')

	results = {
		name: trainer.xtract_ftr(name, **kws)
		for name in dataloader_keys
	}

	times = results['trn']['times']
	times = {i: t for i, t in enumerate(times)}

	ground_truths = {
		k: getattr(trainer, f"dl_{k}")
		for k in dataloader_keys
	}
	ground_truths = {
		k: tonp(dl.dataset.tensors[1]) for
		k, dl in ground_truths.items()
	}
	if 'BALLS' in trainer.model.cfg.dataset:
		ground_truths = {
			k: g[:, [1, 3]] for
			k, g in ground_truths.items()
		}

	return results, times, ground_truths


def _u2z(u, trainer):
	if trainer.model.cfg.type == 'poisson':
		z = np.exp(u)
	else:
		z = tonp(trainer.model.apply_act_fn(
			trainer.to(u)))
	return z
