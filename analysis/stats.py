from utils.generic import *
from statsmodels.stats.multitest import multipletests


def mu_and_err(
		data: np.ndarray,
		dof: int = None,
		fmt: str = '0.1f',
		n_resamples: int = int(1e6),
		ci: float = 0.99, ):

	data = data[np.isfinite(data)]
	assert data.ndim == 1

	if dof is None:
		dof = len(data) - 1

	mu = np.nanmean(data)
	sd = np.nanstd(data)

	cond_skip = (
		sd == 0 or
		dof == 0 or
		len(data) == 1
	)
	if cond_skip:
		err = 0
	else:
		se = sp_stats.bootstrap(
			data=(data,),
			n_resamples=n_resamples,
			statistic=np.nanmean,
			method='BCa',
		).standard_error
		err = se * get_tval(
			dof=dof, ci=ci)

	# make table entry
	mu_str = f"{mu:{fmt}}"
	err_str = f"{err:{fmt}}"
	mu_str, err_str = map(
		_strip0,
		[mu_str, err_str],
	)
	mu_str, err_str = map(
		_add_curly_btacket,
		[mu_str, err_str],
	)
	entry = f"\entry{mu_str}{err_str}"

	return entry, mu, err


def _add_curly_btacket(s: str):
	return f"{'{'}{s}{'}'}"


def _strip0(s: str):
	if s.startswith('0'):
		return s.lstrip('0')
	return s
