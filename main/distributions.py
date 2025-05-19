from base.utils_model import *
dists.Distribution.set_default_validate_args(False)


class Poisson:
	def __init__(
			self,
			log_rate: torch.Tensor,
			temp: float = 0.0,
			clamp: float | None = None,
			n_exp: int | str = 'infer',
			n_exp_p: float = 1e-3,
	):
		assert temp >= 0.0, f"must be non-neg: {temp}"
		self.temp = temp
		self.clamp = clamp
		# setup rate & exp dist
		if clamp is not None:
			log_rate = softclamp_upper(
				log_rate, clamp)
		eps = torch.finfo(torch.float32).eps
		self.rate = torch.exp(log_rate) + eps
		self.exp = dists.Exponential(self.rate)
		# compute n_exp
		if n_exp == 'infer':
			max_rate = self.rate.max().item()
			n_exp = compute_n_exp(max_rate, n_exp_p)
		self.n_exp = int(n_exp)

	@property
	def mean(self):
		return self.rate

	@property
	def variance(self):
		return self.rate

	# noinspection PyTypeChecker
	def rsample(self):
		if self.temp == 0:
			return self.sample()
		x = self.exp.rsample((self.n_exp,))   # inter-event times
		times = torch.cumsum(x, dim=0)        # arrival t of events
		indicator = torch.sigmoid(            # events within [0, 1]
			(1 - times) / self.temp)
		z = indicator.sum(0).float()          # event counts
		return z

	@torch.no_grad()
	def sample(self):
		return torch.poisson(self.rate).float()

	def log_prob(self, samples: torch.Tensor):
		return (
			- self.rate
			- torch.lgamma(samples + 1)
			+ samples * torch.log(self.rate)
		)


# noinspection PyAbstractClass
class Normal(dists.Normal):
	def __init__(
			self,
			loc: torch.Tensor,
			log_scale: torch.Tensor,
			temp: float = 1.0,
			clamp: float | None = None,
			**kwargs,
	):
		if clamp is not None:
			log_scale = softclamp_sym(
				log_scale, clamp)
		super(Normal, self).__init__(
			loc=loc, scale=torch.exp(log_scale), **kwargs)

		assert temp >= 0
		if temp != 1.0:
			self.scale *= temp
		self.temp = temp
		self.clamp = clamp

	def kl(self, p: dists.Normal = None):
		if p is None:
			term1 = self.mean
			term2 = self.scale
		else:
			term1 = (self.mean - p.mean) / p.scale
			term2 = self.scale / p.scale
		kl = 0.5 * (
			term1.pow(2) + term2.pow(2) +
			torch.log(term2).mul(-2) - 1
		)
		return kl

	@torch.no_grad()
	def retrieve_noise(self, samples):
		return (samples - self.loc).div(self.scale)


# noinspection PyAbstractClass
class Laplace(dists.Laplace):
	def __init__(
			self,
			loc: torch.Tensor,
			log_scale: torch.Tensor,
			temp: float = 1.0,
			clamp: float = 5.3,
			**kwargs,
	):
		if clamp is not None:
			log_scale = softclamp_sym(log_scale, clamp)
		super(Laplace, self).__init__(
			loc=loc, scale=torch.exp(log_scale), **kwargs)

		assert temp >= 0
		if temp != 1.0:
			self.scale *= temp
		self.t = temp
		self.c = clamp

	def kl(self, p: dists.Laplace = None):
		if p is not None:
			mean, scale = p.mean, p.scale
		else:
			mean, scale = 0, 1

		delta_m = torch.abs(self.mean - mean)
		delta_b = self.scale / scale
		term1 = delta_m / self.scale
		term2 = delta_m / scale

		kl = (
			delta_b * torch.exp(-term1) +
			term2 - torch.log(delta_b) - 1
		)
		return kl


# noinspection PyAbstractClass
class Categorical(dists.RelaxedOneHotCategorical):
	def __init__(
			self,
			logits: torch.Tensor,
			temp: float = 1.0,
			**kwargs,
	):
		self._categorical = None
		temp = max(temp, torch.finfo(torch.float).eps)
		super(Categorical, self).__init__(
			logits=logits, temperature=temp, **kwargs)

	@property
	def t(self):
		return self.temperature

	@property
	def mean(self):
		return self.probs

	@property
	def variance(self):
		return self.probs * (1 - self.probs)

	def kl(self, p: dists.Categorical = None):
		if p is None:
			probs = torch.full(
				size=self.probs.size(),
				fill_value=1 / self.probs.size(-1),
			)
			p = dists.Categorical(probs=probs)
		q = dists.Categorical(probs=self.probs)
		return dists.kl.kl_divergence(q, p)


def compute_n_exp(rate: float, p: float = 1e-6):
	assert rate > 0.0, f"must be positive, got: {rate}"
	pois = sp_stats.poisson(rate)
	n_exp = pois.ppf(1.0 - p)
	return int(n_exp)


def softclamp(x: torch.Tensor, upper: float, lower: float = 0.0):
	return lower + F.softplus(x - lower) - F.softplus(x - upper)


def softclamp_sym(x: torch.Tensor, c: float):
	return x.div(c).tanh_().mul(c)


def softclamp_sym_inv(y: torch.Tensor, c: float) -> torch.Tensor:
	y = y.clone().detach()

	if not torch.all((y > -c) & (y < c)):
		msg = "must: all(-c < y < c)"
		raise ValueError(msg)

	x = y.div(c).atanh_().mul(c)
	return x


def softclamp_upper(x: torch.Tensor, c: float):
	return c - F.softplus(c - x)


def softclamp_upper_inv(y: torch.Tensor, c: float):
	y = y.clone().detach()

	if not torch.all(y < c):
		msg = "must: all(y < c)"
		raise ValueError(msg)

	a = (c - y).float()
	log_term = torch.log1p(-torch.exp(-a))
	log_expm1 = a + log_term
	x = c - log_expm1
	return x


@torch.jit.script
def sample_normal_jit(
		mu: torch.Tensor,
		sigma: torch.Tensor, ):
	eps = torch.empty_like(mu).normal_()
	return sigma * eps + mu


def sample_normal(
		mu: torch.Tensor,
		sigma: torch.Tensor,
		rng: torch.Generator = None, ):
	eps = torch.empty_like(mu).normal_(
		mean=0., std=1., generator=rng)
	return sigma * eps + mu


@torch.jit.script
def residual_kl(
		delta_mu: torch.Tensor,
		delta_sig: torch.Tensor,
		sigma: torch.Tensor, ):
	return 0.5 * (
		delta_sig.pow(2) - 1 +
		(delta_mu / sigma).pow(2)
	) - torch.log(delta_sig)
