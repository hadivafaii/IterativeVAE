from utils.plotting import *


def plot_weights(
		w: np.ndarray | torch.Tensor,
		nrows: int = None,
		method: str = 'min-max',
		title: str = None,
		display: bool = True,
		**kwargs, ):
	defaults = dict(
		dpi=160,
		figsize=(8, 4),
		title_fontsize=8,
		title_y=1.01,
		vmin=None,
		vmax=None,
		cmap=None,
	)
	kwargs = setup_kwargs(defaults, kwargs)

	# infer nrows
	if nrows is None:
		a = np.log2(len(w))
		a = int(np.ceil(a))
		if a % 2 == 1:
			a += 1
		if a <= 6:
			exponent = a // 3 - 1
		elif len(w) == 128:
			exponent = 2
		else:
			exponent = a // 2 - 1
		nrows = int(2 ** exponent)
		nrows = max(1, nrows)

	# make grid
	kws_grid = filter_kwargs(make_grid, kwargs)
	ncols = int(np.ceil(len(w) / nrows))
	grid_dec = make_grid(
		x=w,
		grid_size=(nrows, ncols),
		normalize=False if
		method == 'none'
		else True,
		method=method,
		**kws_grid,
	)
	# plot
	fig, ax = create_figure(
		figsize=kwargs['figsize'],
		dpi=kwargs['dpi'],
		layout='tight',
	)
	if method == 'min-max':
		vmin, vmax = 0, 1
		cmap = kwargs['cmap'] or 'Greys_r'
	elif method == 'abs-max':
		vmin, vmax = -1, 1
		cmap = kwargs['cmap'] or 'Greys_r'
	elif method == 'none':
		vmin, vmax = None, None
		cmap = kwargs['cmap'] or 'Greys_r'
	else:
		raise ValueError(method)
	kws_show = dict(
		vmin=kwargs['vmin'] or vmin,
		vmax=kwargs['vmax'] or vmax,
		cmap=kwargs['cmap'] or cmap,
	)
	ax.imshow(grid_dec, **kws_show)
	ax.set_title(
		label=title,
		fontsize=kwargs['title_fontsize'],
		y=kwargs['title_y'],
	)
	remove_ticks(ax)
	if display:
		plt.show()
	else:
		plt.close()
	return fig, ax


def make_grid(
		x: np.ndarray | torch.Tensor,
		grid_size: int | Tuple[int, int],
		scaling: Sequence[float] = None,
		pad: int = 1,
		pad_val: float = np.nan,
		normalize: bool = True,
		**kwargs, ):
	x = tonp(x)
	if x.ndim == 3:
		x = x[:, np.newaxis]
	assert x.ndim == 4
	x = x.transpose(0, 2, 3, 1)
	b, h, w, c = x.shape

	if scaling is None:
		scaling = [1.0] * b
	assert len(scaling) == b

	if isinstance(grid_size, int):
		grid_size = (grid_size, grid_size)
	n_rows, n_cols = grid_size

	grid = np.ones((
		(h + pad) * n_rows - pad,
		(w + pad) * n_cols - pad,
		c,
	)) * pad_val

	for idx in range(min(n_rows * n_cols, b)):
		i = idx // n_cols
		j = idx % n_cols
		a = (h + pad) * i
		b = (w + pad) * j

		y = x[idx]
		if normalize:
			y = normalize_img(
				y, **kwargs)
		y *= scaling[idx]  # apply manual scaling
		grid[a:a + h, b:b + w] = y

	return grid


def normalize_img(
		x: np.ndarray,
		method: str = 'min-max',
		val_range: Tuple[float, float] = (0, 1), ):
	if method == 'min-max':
		xmin = np.min(x)
		xmax = np.max(x)

		numen = x - xmin
		denum = xmax - xmin
		x_nrm = numen / denum

		a, b = min(val_range), max(val_range)
		x_nrm = x_nrm * (b - a) + a

	elif method == 'abs-max':
		x_nrm = x / np.max(np.abs(x))
	else:
		raise ValueError(method)

	return x_nrm
