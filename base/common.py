from .utils_model import *
MULT = 2


def get_stride(cell_type: str, cmult: int):
	startswith = cell_type.split('_')[0]
	if startswith in ['normal', 'combiner']:
		stride = 1
	elif startswith == 'down':
		stride = cmult
	elif startswith == 'up':
		stride = -1
	else:
		raise NotImplementedError(cell_type)
	return stride


def get_skip_connection(
		ci: int,
		cmult: int,
		stride: Union[int, str],
		reg_lognorm: bool = True,
		factorized: bool = True, ):
	if isinstance(stride, str):
		stride = get_stride(stride, cmult)
	if stride == 1:
		return nn.Identity()
	elif stride in [2, 4]:
		kws = dict(
			ci=ci,
			co=int(cmult*ci),
			reg_lognorm=reg_lognorm,
		)
		if factorized:
			return FactorizedReduce(**kws)
		else:
			return StrideReduce(**kws)
	elif stride == -1:
		return nn.Sequential(
			nn.Upsample(
				scale_factor=cmult,
				mode='nearest'),
			Conv2D(
				kernel_size=1,
				in_channels=ci,
				out_channels=int(ci/cmult),
				reg_lognorm=reg_lognorm),
		)
	else:
		raise NotImplementedError(stride)


def get_act_fn(
		fn: str,
		inplace: bool = False,
		**kwargs, ):
	if fn == 'none':
		return None
	elif fn == 'relu':
		return nn.ReLU(inplace=inplace)
	elif fn == 'leaky_relu':
		return nn.LeakyReLU(inplace=inplace, **kwargs)
	elif fn in ['swish', 'silu']:
		return nn.SiLU(inplace=inplace)
	elif fn == 'elu':
		return nn.ELU(inplace=inplace, **kwargs)
	elif fn == 'gelu':
		return nn.GELU(**kwargs)
	elif fn == 'softplus':
		return nn.Softplus(**kwargs)
	else:
		raise NotImplementedError(fn)


class StrideReduce(nn.Module):
	def __init__(self, ci: int, co: int, **kwargs):
		super(StrideReduce, self).__init__()
		defaults = {
			'kernel_size': 1,
			'in_channels': ci,
			'out_channels': co,
			'reg_lognorm': True,
			'stride': co // ci,
			'padding': 0,
			'bias': True,
		}
		kwargs = setup_kwargs(defaults, kwargs)
		self.swish = nn.SiLU()
		self.conv = Conv2D(**kwargs)

	def forward(self, x):
		x = self.swish(x)
		x = self.conv(x)
		return x


class FactorizedReduce(nn.Module):
	def __init__(self, ci: int, co: int, **kwargs):
		super(FactorizedReduce, self).__init__()
		assert co % 2 == 0 and co > 4
		co_each = co // 4
		defaults = {
			'kernel_size': 1,
			'in_channels': ci,
			'out_channels': co_each,
			'reg_lognorm': True,
			'stride': co // ci,
			'padding': 0,
			'bias': True,
		}
		kwargs = setup_kwargs(defaults, kwargs)
		self.swish = nn.SiLU()
		self.ops = nn.ModuleList()
		for _ in range(3):
			self.ops.append(Conv2D(**kwargs))
		kwargs['out_channels'] = co - 3 * co_each
		self.ops.append(Conv2D(**kwargs))

	def forward(self, x):
		x = self.swish(x)
		idx, out = 0, []
		for op in self.ops:
			i, j = idx // 2, idx % 2
			out.append(op(x[..., i:, j:]))
			idx += 1
		return torch.cat(out, dim=1)


class SELayer(nn.Module):
	def __init__(self, ci: int, reduc: int = 16):
		super(SELayer, self).__init__()
		self.hdim = max(ci // reduc, 4)
		self.fc = nn.Sequential(
			nn.Linear(ci, self.hdim), nn.ReLU(inplace=True),
			nn.Linear(self.hdim, ci), nn.Sigmoid(),
		)

	def forward(self, x):
		b, c, _, _ = x.size()
		se = torch.mean(x, dim=[2, 3])
		se = self.fc(se).view(b, c, 1, 1)
		return x * se


class Cell(nn.Module):
	def __init__(
			self,
			ci: int,
			co: int,
			cell_type: str,
			n_nodes: int,
			act_fn: str,
			use_bn: bool,
			use_se: bool,
			scale: float,
			eps: float,
			**kwargs,
	):
		super(Cell, self).__init__()
		assert n_nodes >= 1
		kws_skip = filter_kwargs(
			get_skip_connection, kwargs)
		self.skip = get_skip_connection(
			ci, MULT, cell_type, **kws_skip)
		self.ops = nn.ModuleList()
		for i in range(n_nodes):
			op = ConvLayer(
				ci=ci if i == 0 else co,
				co=co,
				stride=get_stride(cell_type, MULT)
				if i == 0 else 1,
				act_fn=act_fn,
				use_bn=use_bn,
				init_gain=scale
				if i+1 == n_nodes
				else 1.0,
				**kwargs,
			)
			self.ops.append(op)
		if use_se:
			self.se = SELayer(co)
		else:
			self.se = None
		self.eps = eps

	def forward(self, x):
		skip = self.skip(x)
		for op in self.ops:
			x = op(x)
		if self.se is not None:
			x = self.se(x)
		return skip + self.eps * x


class ConvLayer(nn.Module):
	def __init__(
			self,
			ci: int,
			co: int,
			stride: int,
			act_fn: str,
			use_bn: bool,
			**kwargs,
	):
		super(ConvLayer, self).__init__()
		defaults = {
			'in_channels': ci,
			'out_channels': co,
			'kernel_size': 3,
			'normalize_dim': 0,
			'reg_lognorm': True,
			'init_scale': 1.0,
			'stride': abs(stride),
			'padding': 1,
			'dilation': 1,
			'groups': 1,
			'bias': True,
		}
		if stride == -1:
			self.upsample = nn.Upsample(
				scale_factor=MULT,
				mode='nearest',
			)
		else:
			self.upsample = None
		if use_bn:
			self.bn = nn.BatchNorm2d(ci)
		else:
			self.bn = None
		self.act_fn = get_act_fn(act_fn, False)
		kwargs = setup_kwargs(defaults, kwargs)
		self.conv = Conv2D(**kwargs)

	def forward(self, x):
		if self.bn is not None:
			x = self.bn(x)
		if self.act_fn is not None:
			x = self.act_fn(x)
		if self.upsample is not None:
			x = self.upsample(x)
		x = self.conv(x)
		return x


class _NormalizableLayer(nn.Module):
	def __init__(
			self,
			normalize_dim: int,
			lognorm_size: tuple,
			normalize: bool = False,
			fit_gain: bool = False,
			init_gain: float = 1.0,
	):
		super(_NormalizableLayer, self).__init__()
		assert init_gain > 0
		self.normalize = normalize
		self.normalize_dim = normalize_dim
		self.lognorm = None
		if fit_gain:
			self._init_lognorm(
				lognorm_size, init_gain)
		self._layer = None

	def get_weight(self):
		if self.normalize:
			w = F.normalize(
				input=self.weight,
				dim=self.normalize_dim,
			)
		else:
			w = self.weight
		if self.lognorm is not None:
			return w * torch.exp(self.lognorm)
		return w

	def _init_lognorm(self, size, init_gain):
		init = torch.ones(size).mul(init_gain)
		self.lognorm = nn.Parameter(
			data=torch.log(init),
			requires_grad=True,
		)
		return

	@property
	def weight(self):
		return self._layer.weight

	@property
	def bias(self):
		return self._layer.bias


class Linear(_NormalizableLayer):
	def __init__(
			self,
			in_features: int,
			out_features: int,
			normalize_dim: int = 0,
			**kwargs,
	):
		super(Linear, self).__init__(
			normalize_dim=normalize_dim,
			lognorm_size=(out_features, 1),
			**filter_kwargs(_NormalizableLayer, kwargs),
		)
		self._layer = nn.Linear(
			in_features=in_features,
			out_features=out_features,
			**filter_kwargs(nn.Linear, kwargs),
		)

	def forward(self, x):
		return F.linear(
			input=x,
			weight=self.get_weight(),
			bias=self._layer.bias,
		)


class Conv1D(_NormalizableLayer):
	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			kernel_size: int,
			normalize_dim: int = 0,
			**kwargs,
	):
		super(Conv1D, self).__init__(
			normalize_dim=normalize_dim,
			lognorm_size=(out_channels, 1, 1),
			**filter_kwargs(_NormalizableLayer, kwargs),
		)
		self.pad = kernel_size - 1
		kwargs['padding'] = self.pad
		self._layer = nn.Conv1d(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=kernel_size,
			**filter_kwargs(nn.Conv1d, kwargs),
		)

	def forward(self, x):
		return F.conv1d(
			input=x,
			weight=self.get_weight(),
			bias=self._layer.bias,
			stride=self._layer.stride,
			padding=self._layer.padding,
			dilation=self._layer.dilation,
			groups=self._layer.groups,
		)[..., :-self.pad].contiguous()


class Conv2D(_NormalizableLayer):
	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			kernel_size: Union[int, Tuple[int, int]],
			normalize_dim: int = 0,
			**kwargs,
	):
		super(Conv2D, self).__init__(
			normalize_dim=normalize_dim,
			lognorm_size=(out_channels, 1, 1, 1),
			**filter_kwargs(_NormalizableLayer, kwargs),
		)
		self._layer = nn.Conv2d(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=kernel_size,
			**filter_kwargs(nn.Conv2d, kwargs),
		)

	def forward(self, x):
		return F.conv2d(
			input=x,
			weight=self.get_weight(),
			bias=self._layer.bias,
			stride=self._layer.stride,
			padding=self._layer.padding,
			dilation=self._layer.dilation,
			groups=self._layer.groups,
		)


class DeConv2D(_NormalizableLayer):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		kernel_size: Union[int, Tuple[int, int]],
		normalize_dim: int = 1,
		**kwargs,
	):
		super(DeConv2D, self).__init__(
			normalize_dim=normalize_dim,
			lognorm_size=(1, out_channels, 1, 1),
			**filter_kwargs(_NormalizableLayer, kwargs),
		)
		self._layer = nn.ConvTranspose2d(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=kernel_size,
			**filter_kwargs(nn.ConvTranspose2d, kwargs),
		)

	def forward(self, x, output_size=None):
		return F.conv_transpose2d(
			input=x,
			weight=self.get_weight(),
			bias=self._layer.bias,
			stride=self._layer.stride,
			padding=self._layer.padding,
			dilation=self._layer.dilation,
			groups=self._layer.groups,
		)


class RotConv2d(nn.Conv2d):
	def __init__(
			self,
			co: int,
			n_rots: int,
			kernel_size: Union[int, Iterable[int]],
			bias: bool = True,
			gain: bool = True,
			**kwargs,
	):
		super(RotConv2d, self).__init__(
			in_channels=2,
			out_channels=co,
			kernel_size=kernel_size,
			padding='valid',
			bias=bias,
			**kwargs,
		)
		self.n_rots = n_rots
		self._build_rot_mat()
		if bias:
			bias = nn.Parameter(
				torch.zeros(co*n_rots),
				requires_grad=True,
			)
		else:
			bias = None
		self.bias = bias
		if gain:
			gain = nn.Parameter(
				torch.ones(co*n_rots),
				requires_grad=True,
			)
		else:
			gain = None
		self.gain = gain
		self.w = self.augment_weight()

	def forward(self, x):
		self.w = self.augment_weight()
		return F.conv2d(
			input=x,
			weight=self.w,
			bias=self.bias,
			stride=self.stride,
			padding=self.padding,
			dilation=self.dilation,
			groups=self.groups,
		)

	def _build_rot_mat(self):
		thetas = np.deg2rad(np.arange(
			0, 360, 360 / self.n_rots))
		u = [0.0, 0.0, 1.0]
		u = np.array(u).reshape(1, -1)
		u = np.repeat(u, self.n_rots, 0)
		u *= thetas.reshape(-1, 1)
		r = Rotation.from_rotvec(u)
		r = r.as_matrix()
		r = torch.tensor(
			data=r[:, :2, :2],
			dtype=torch.float,
		)
		self.register_buffer('rot_mat', r)
		return

	def augment_weight(self):
		eps = torch.finfo(torch.float32).eps
		wn = torch.linalg.vector_norm(
			x=self.weight,
			dim=[1, 2, 3],
			keepdim=True,
		)
		w = torch.einsum(
			'rij, kjxy -> krixy',
			self.rot_mat,
			self.weight / (wn + eps),
		).flatten(end_dim=1)
		if self.gain is not None:
			w *= self.gain.view(
				-1, 1, 1, 1)
		return w


class UnFlatten(nn.Module):
	def __init__(self, n_channels: int):
		super(UnFlatten, self).__init__()
		self.n_channels = n_channels

	def forward(self, x):
		if x.ndim == 1:
			x = x.unsqueeze(0)
		s = int((x.size(1) // self.n_channels) ** 0.5)
		return x.view(x.size(0), self.n_channels, s, s)


class AddNorm(object):
	def __init__(self, norm, types, **kwargs):
		super(AddNorm, self).__init__()
		self.norm = norm
		self.types = types
		if self.norm == 'spectral':
			self.kwargs = filter_kwargs(
				fn=nn.utils.parametrizations.spectral_norm,
				kw=kwargs,
			)
		elif self.norm == 'weight':
			self.kwargs = filter_kwargs(
				fn=nn.utils.weight_norm,
				kw=kwargs,
			)
		else:
			raise NotImplementedError

	def get_fn(self) -> Callable:
		if self.norm == 'spectral':
			def fn(m):
				if isinstance(m, self.types):
					nn.utils.parametrizations.spectral_norm(
						module=m, **self.kwargs)
				return
		elif self.norm == 'weight':
			def fn(m):
				if isinstance(m, self.types):
					nn.utils.weight_norm(
						module=m, **self.kwargs)
				return
		else:
			raise NotImplementedError
		return fn
