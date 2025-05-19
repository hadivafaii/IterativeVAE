from .generic import *
from .plotting import plt


def xtract_patches_random(
		dataset,
		npix: int,
		n_patches: int,
		batch_size: int = 500,
		verbose: bool = False,
		seed: int = 0, ):
	loader = tqdm(
		iterable=iter(torch.utils.data.DataLoader(
			dataset=dataset,
			batch_size=batch_size,
			shuffle=False)),
		disable=not verbose,
		ncols=80,
	)
	rng = get_rng(seed)
	patches = []
	for x, *_ in loader:
		n, _, h, w = x.shape

		tt = rng.integers(0, h - npix, size=(n, n_patches))
		ll = rng.integers(0, w - npix, size=(n, n_patches))

		# Vectorized extraction of patches
		patches_batch = torch.stack([
			x[i, :, tt[i, j]:tt[i, j] + npix, ll[i, j]:ll[i, j] + npix]
			for i in range(n) for j in range(n_patches)
		])
		patches.append(patches_batch)

	patches = torch.cat(patches)
	return patches


def xtract_patches(x: torch.Tensor, npix: int):
	b, c = x.shape[:2]
	dims = (npix, npix)
	patches = x.unfold(2, *dims).unfold(3, *dims)
	patches = patches.contiguous().view(b, -1, c, *dims)
	return patches


def do_process(
		stim: torch.Tensor,
		crop_size: int = 4,
		kws_whiten: dict = None,
		kws_contrast: dict = None, ):
	if kws_whiten is None:
		kws_whiten = dict(
			f_0=0.5,
			n=4,
			batched=True,
		)
	if kws_contrast is None:
		kws_contrast = dict(
			kernel_size=13,
			sigma=0.5,
			batched=True,
		)

	whitener = Whitening(tuple(stim.shape[-2:]), **kws_whiten)
	contrast_ner = LocContrastNorm(**kws_contrast)

	if stim.ndim == 4:
		x_wt = whitener(stim)
		x_wt_cn = contrast_ner(x_wt)
		x_wt_cn_zs = zscore(x_wt_cn)
	elif stim.ndim == 5:
		x_wt = []
		x_wt_cn = []
		x_wt_cn_zs = []
		for _x in stim:
			_xwt = whitener(_x.cuda())
			_xwtcn = contrast_ner(_xwt)
			x_wt.append(_xwt)
			x_wt_cn.append(_xwtcn)
			x_wt_cn_zs.append(zscore(_xwtcn))
		x_wt, x_wt_cn, x_wt_cn_zs = map(
			lambda t: torch.stack(t),
			[x_wt, x_wt_cn, x_wt_cn_zs],
		)
	else:
		raise RuntimeError(f"incompatible stim shape: {stim.shape}")

	c = crop_size
	x_wt, x_wt_cn, x_wt_cn_zs = map(
		lambda t: t[..., c:-c, c:-c],
		[x_wt, x_wt_cn, x_wt_cn_zs],
	)
	return x_wt, x_wt_cn, x_wt_cn_zs


def compute_fftfreq(npix):
	freq = np.fft.fftfreq(npix) * npix
	a, b = np.meshgrid(freq, freq)
	freq_norm = np.sqrt(a**2 + b**2)

	bins = np.arange(0.5, npix//2 + 1, 1.0)
	kvals = 0.5 * (bins[1:] + bins[:-1])

	return freq_norm, kvals, bins


def do_fft(x):
	x_ft = torch.fft.fft2(x).squeeze()
	# phase = torch.angle(x_ft)
	amplitude = torch.abs(x_ft)
	power = tonp(amplitude ** 2)

	freq_norm, kvals, bins = compute_fftfreq(x.size(-1))

	psd = np.zeros((len(x), len(kvals)))
	for i in range(len(x)):
		p, _, _ = sp_stats.binned_statistic(
			x=freq_norm.ravel(),
			values=power[i].ravel(),
			statistic='mean',
			bins=bins,
		)
		p *= np.pi * (bins[1:]**2 - bins[:-1]**2)
		psd[i] = p
	return psd


class LocContrastNorm(object):
	"""
	Local Contrast Normalization as defined in Jarret et al. 2009
	(http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf)
	"""

	def __init__(self, kernel_size=9, sigma=0.5, rgb=False, batched=False):
		self.batched = batched
		if kernel_size % 2 == 0:
			raise RuntimeError('kernel size must be odd...')
		x = torch.from_numpy(np.linspace(-1, 1, kernel_size))
		x = x.unsqueeze(1).expand(kernel_size, kernel_size).float()
		y = torch.from_numpy(np.linspace(-1, 1, kernel_size))
		y = y.unsqueeze(0).expand(kernel_size, kernel_size).float()
		r_2 = x.pow(2) + y.pow(2)
		self.kernel_size = kernel_size
		self.gaussian_k = torch.exp(-r_2/sigma).unsqueeze(0).unsqueeze(0)
		if rgb:
			self.gaussian_k = self.gaussian_k.expand(
				3, 3, kernel_size, kernel_size)
		self.gaussian_k = self.gaussian_k / self.gaussian_k.sum()
		if torch.cuda.is_available():
			self.gaussian_k = self.gaussian_k.cuda()

	def __call__(self, img):
		# subtractive step
		if not self.batched:
			img = img.unsqueeze(0)
		img_pad = F.pad(img, ((self.kernel_size - 1) // 2, ) * 4)
		gaus_map = F.conv2d(img_pad, self.gaussian_k)
		img_sub = img - gaus_map
		# divisive step
		img_pad = F.pad(img_sub, ((self.kernel_size - 1) // 2, ) * 4)
		img_sigma = F.conv2d(img_pad.pow(2), self.gaussian_k).sqrt()
		c = img_sigma.view(img_sigma.size()[0], -1).mean(-1, keepdim=True)
		c = c.unsqueeze(-1).unsqueeze(-1).expand_as(img_sigma)
		img_sigma = (
			(F.relu(img_sigma - c) > 0).float() * img_sigma +
			(1 - (F.relu(img_sigma - c) > 0).float()) * c
		)
		img_div = img_sub / img_sigma

		if not self.batched:
			return img_div.squeeze(0)
		return img_div


class ZScore(object):
	"""
	Image per Image z-score normalization
	Image = (Image-mean(Image))/std(Image)
	"""

	def __call__(self, img):
		img = img - img.mean()
		img = img / img.std()
		return img


class Whitening(object):
	"""
	Whitening filter as in Olshausen&Field 1995
	R(f) = f * exp((f/f_0)^n)
	"""

	def __init__(self, img_size, f_0=0.5, n=4, batched=False):
		self.f_0 = f_0
		self.batched = batched
		self.n = n
		# if not torch.backends.mkl.is_available():
		# 	raise Exception('MKL not found')
		dim_x = img_size[0]
		dim_y = img_size[1]
		f_x = torch.from_numpy(np.linspace(-0.5, 0.5, dim_x))
		f_x = f_x.unsqueeze(1).expand(dim_x, dim_y).float()
		f_y = torch.from_numpy(np.linspace(-0.5, 0.5, dim_y))
		f_y = f_y.unsqueeze(0).expand(dim_x, dim_y).float()
		f = (f_x.pow(2) + f_y.pow(2)).sqrt()
		self.f = f.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
		if torch.cuda.is_available():
			self.f = self.f.cuda()

	def __call__(self, img, verbose=False):
		if not self.batched:
			img = img.unsqueeze(0)
		img_f = torch.fft.fft2(img)  # , onesided=False)
		filt = self.f * torch.exp(-(self.f / self.f_0).pow(self.n))
		img_f_ = img_f * filt.squeeze(-1)
		if verbose:
			plt.imshow(filt[0, 0, :, :, 0].cpu().numpy())
			plt.colorbar()
			plt.show()
		img = torch.fft.ifft2(img_f_).real  # , onesided=False)
		if not self.batched:
			return img.squeeze(0)
		return img


class EyeDataset(object):
	def __init__(
			self,
			grid_w: int = 64,
			grid_h: int = 64,
			device: str = 'cuda:0',
			traces_file: str = 'data1024.h5',
			imgs_file: str = 'vHimgs.h5',
			path: str = 'Datasets/DOVES',
	):
		super().__init__()
		self.path = pjoin(os.environ['HOME'], path)
		self.traces_file = traces_file
		self.imgs_file = imgs_file
		self.device = device
		self._load()
		self.w, self.h = grid_w, grid_h

	def _load(self):
		with h5py.File(pjoin(self.path, self.traces_file), 'r') as f:
			trajs = f['eye_traces'][()].T

		with h5py.File(pjoin(self.path, self.imgs_file), 'r') as f:
			imgs = f['imgs'][()].T

		self.trajs = torch.Tensor(trajs).to(self.device)  # [N, T, 2]
		self.imgs = torch.Tensor(imgs).to(self.device).unsqueeze(1)  # [B, C, H, W]
		return

	def build(
			self,
			ids_img: Iterable[int],
			ids_traj: Iterable[int],
			verbose: bool = False, ):

		if isinstance(ids_img, int):
			ids_img = [ids_img]
		if isinstance(ids_traj, int):
			ids_traj = [ids_traj]

		looper = tqdm(
			iterable=itertools.product(
				ids_traj, ids_img),
			total=len(ids_img) * len(ids_traj),
			disable=not verbose,
		)
		stim = []
		for t, i in looper:
			traj = self.trajs[t, :, :]
			img = self.imgs[[i], :, :, :]
			stim.append(self._grid_sample(
				traj, img, self.trajs.shape[1]))

		return torch.stack(stim)

	def _grid_sample(self, traj, img, tlim):
		im_center = [img.shape[3] / 2.0, img.shape[2] / 2.0]
		dx = torch.linspace(-1, 1, int(self.w))
		dy = torch.linspace(-1, 1, int(self.h))
		meshy, meshx = torch.meshgrid(dy, dx, indexing="ij")
		grid = torch.stack((meshx, meshy), 2).to(self.device)

		img_sample = torch.zeros(tlim, 1, self.w, self.h)
		for i in range(tlim):
			grid_center = torch.Tensor([traj[i, 0], traj[i, 1]])

			# Shift and scale the grid (width x height)
			w = (grid_center[0] - im_center[0]) / im_center[0]
			h = (grid_center[1] - im_center[1]) / im_center[1]

			g = grid / 15
			g[..., 0] += w
			g[..., 1] += h

			sample = F.grid_sample(
				img, g.unsqueeze(0), align_corners=False)
			img_sample[i, :, :, :] = sample
		return img_sample


def zscore(x, dim=(-2, -1)):
	mu = torch.mean(x, dim=dim, keepdim=True)
	sd = torch.std(x, dim=dim, keepdim=True)
	return (x - mu) / sd


def filter_boundaries(
		x_list: Union[np.ndarray, List[np.ndarray]],
		mask: np.ndarray = None, ):
	if not isinstance(x_list, list):
		x_list = [x_list]
	if mask is None:
		# noinspection PyTypeChecker
		mask = functools.reduce(
			np.logical_or,
			[x.astype(bool) for x in x_list]
		)
	dim = len(mask.shape)

	if dim == 2:
		bad_x_indices = np.where(mask.mean(1) == 0)[0]
		bad_y_indices = np.where(mask.mean(0) == 0)[0]
		bad_z_indices = []
	elif dim == 3:
		bad_x_indices = np.where(mask.mean(-1).mean(-1) == 0)[0]
		bad_y_indices = np.where(mask.mean(0).mean(-1) == 0)[0]
		bad_z_indices = np.where(mask.mean(0).mean(0) == 0)[0]
	else:
		raise ValueError("wrong dim: {}".format(dim))

	filtered = []
	for x in x_list:
		if dim == 2:
			x_filtered = np.delete(np.delete(
				x, bad_y_indices, axis=1),
				bad_x_indices, axis=0,
			)
		elif dim == 3:
			x_filtered = np.delete(np.delete(np.delete(
				arr=x, obj=bad_z_indices, axis=2,
			), obj=bad_y_indices, axis=1,
			), obj=bad_x_indices, axis=0,
			)
		else:
			raise ValueError("wrong dim: {}".format(dim))
		filtered.append(x_filtered)
	return filtered if len(filtered) > 1 else filtered.pop()
