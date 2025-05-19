from utils.generic import *


def sparse_score(
		z: Union[np.ndarray, torch.Tensor],
		cutoff: float = None, ):
	def _compute_score(axis: int, fix: bool = True):
		m = z.shape[axis]
		lib = np if isinstance(z, np.ndarray) else torch
		numen = lib.sum(z, axis=axis) ** 2
		denum = lib.sum(z ** 2, axis=axis)
		mask = (numen == 0) & (denum == 0)  # no spiks
		denum[denum == 0] = lib.finfo(lib.float32).eps
		score = 1 - (numen / denum) / m
		score /= (1 - 1 / m)
		if fix:  # no spiks
			score[mask] = 1.0
		return score

	if not isinstance(z, (np.ndarray, torch.Tensor)):
		z = torch.tensor(z).cuda()
	if z.ndim == 1:
		z = z.reshape(-1, 1)

	lifetime = _compute_score(0)
	population = _compute_score(-1)

	# percentages
	if cutoff is None:
		percents = None
	else:
		z = tonp(z).ravel()
		counts = collections.Counter(
			np.round(z).astype(int))
		portions = {
			k: v / np.prod(z.shape) for
			k, v in counts.most_common()
		}
		try:
			cutoff = next(
				k + 1 for k, v in
				portions.items()
				if v < cutoff
			)
		except StopIteration:
			cutoff = np.inf
		percents = {
			str(k): v for k, v
			in portions.items()
			if k < cutoff
		}
		percents[f'{cutoff}+'] = sum(
			v for k, v in
			portions.items()
			if k >= cutoff
		)
		percents = {
			k: np.round(v * 100, 1) for
			k, v in percents.items()
		}
		percents = dict(sorted(
			percents.items(),
			key=lambda t: _sort_key(t[0])
		))
	return lifetime, population, percents


def _sort_key(key):
	match = re.match(r"(\d+)(\+?)", key)
	if match:
		num, plus = match.groups()
		return int(num), plus == '+'
	return float('inf'), False
