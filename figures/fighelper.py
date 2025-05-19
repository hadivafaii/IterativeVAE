from utils.plotting import *


def get_betas(df: pd.DataFrame):
	betas: List[Union[float, str]] = ['ae']
	betas += sorted([
		b for b in df['beta'].unique()
		if isinstance(b, float)
	])
	return betas


def get_palette_cube():
	kws_1 = dict(start=2.6, rot=0.03, light=0.7, dark=0.3)
	kws_2 = dict(start=0.6, rot=0.03, light=0.7, dark=0.3)
	cube_1 = get_cubehelix_palette(n_colors=3, **kws_1)
	cube_2 = get_cubehelix_palette(n_colors=3, **kws_2)
	return cube_1, cube_2


def get_marker_style(seq_len: int):
	if seq_len == -1:    # plus (LCA)
		marker = 'P'
		fillstyle = 'full'
	elif seq_len == 1:       # empty square (amort)
		marker = 's'
		fillstyle = 'none'
	elif seq_len == 8:     # triangle
		marker = '^'
		fillstyle = 'full'
	elif seq_len == 16:    # cross
		marker = 'X'
		fillstyle = 'full'
	elif seq_len == 32:    # circle
		marker = 'o'
		fillstyle = 'full'
	else:
		msg = f"unavailable: {seq_len}"
		raise ValueError(msg)

	return marker, fillstyle


def get_palette_models(pal: str = 'tab10'):
	pal = sns.color_palette(pal)
	pal_models = {
		'poisson': pal[0],
		'poisson+amort': pal[0],
		'gaussian': pal[2],
		'gaussian+amort': pal[2],
		'gaussian+relu': pal[3],
		'gaussian+relu+amort': pal[3],
		'lca': pal[7],
		'gaussian+softplus': pal[8],
		'gaussian+softplus+amort': pal[8],
		'gaussian+logistic': pal[4],
		'gaussian+logistic+amort': pal[4],
		'gaussian+gcdf': pal[5],
		'gaussian+gcdf+amort': pal[5],
		'gaussian+tanh': pal[6],
		'gaussian+tanh+amort': pal[6],
		'gaussian+arctan': pal[9],
		'gaussian+arctan+amort': pal[9],
		'gaussian+gompertz': pal[1],
		'gaussian+gompertz+amort': pal[1],
	}
	return pal_models


# noinspection PyTypeChecker
def get_palette():
	# poisson
	betas = [0.01, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 4.0]
	colors = sns.color_palette(
		'crest', n_colors=len(betas))
	palette = {
		'poisson' if b == 1.0 else f"poisson-{b:0.2g}":
			c for b, c in zip(betas, colors)
	}
	palette_beta = {
		b: c for b, c in zip(betas, colors)
	}
	# gaussian
	items = ['-relu', '', '-exp']
	colors = sns.color_palette(
		'flare', n_colors=len(items))
	palette.update({
		'gaussian' + k: c for k, c
		in zip(items, colors)
	})
	# laplace, categorical
	muted = sns.color_palette('muted')
	palette.update({
		'laplace': muted[2],
		'categorical': muted[5],
	})
	# lca, ista
	palette.update({
		'lca': '#6f6f6f',
		'ista': '#aeaeae',
	})
	# lambda (lca)
	lamb = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.7, 1.0]
	colors = sns.cubehelix_palette(n_colors=len(lamb))
	palette_lamb = {
		lamb: c for lamb, c in zip(lamb, colors)
	}
	return palette, palette_beta, palette_lamb


def show_palette(palette: dict, ncols: int = 8):
	nrows = -(-len(palette) // ncols)
	fig, axes = create_figure(
		nrows=nrows,
		ncols=ncols,
		figsize=(ncols * 1.2, nrows * 1.4),
	)
	axes = axes.flatten()

	for i, (key, color) in enumerate(palette.items()):
		axes[i].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
		axes[i].axis('off')
		axes[i].set_title(key)

	for j in range(len(palette), len(axes)):
		axes[j].axis('off')

	plt.show()
	return


def plot_bar(df: pd.DataFrame, display: bool = True, **kwargs):
	defaults = dict(
		x='x',
		y='y',
		figsize_y=7,
		figsize_x=0.7,
		tick_labelsize_x=15,
		tick_labelsize_y=15,
		ylabel_fontsize=20,
		title_fontsize=18,
		vals_fontsize=13,
		title_y=1,
	)
	kwargs = setup_kwargs(defaults, kwargs)
	figsize = (
		kwargs['figsize_x'] * len(df),
		kwargs['figsize_y'],
	)
	fig, ax = create_figure(1, 1, figsize)
	bp = sns.barplot(data=df, x=kwargs['x'], y=kwargs['y'], ax=ax)
	barplot_add_vals(bp, fontsize=kwargs['vals_fontsize'])
	ax.tick_params(
		axis='x',
		rotation=-90,
		labelsize=kwargs['tick_labelsize_x'],
	)
	ax.tick_params(
		axis='y',
		labelsize=kwargs['tick_labelsize_y'],
	)
	val = np.nanmean(df[kwargs['y']]) * 100
	title = r'avg $R^2 = $' + f"{val:0.1f} %"
	ax.set_title(
		label=title,
		y=kwargs['title_y'],
		fontsize=kwargs['title_fontsize'],
	)
	ax.set_ylabel(
		ylabel=r'$R^2$',
		fontsize=kwargs['ylabel_fontsize'],
	)
	ax.set(xlabel='', ylim=(0, 1))
	ax.grid()
	if display:
		plt.show()
	else:
		plt.close()
	return fig, ax


def plot_heatmap(
		r: np.ndarray,
		title: str = None,
		xticklabels: List[str] = None,
		yticklabels: List[str] = None,
		display: bool = True,
		**kwargs, ):
	defaults = dict(
		figsize=(10, 8),
		tick_labelsize_x=12,
		tick_labelsize_y=12,
		title_fontsize=13,
		title_y=1,
		vmin=-1,
		vmax=1,
		cmap='bwr',
		linewidths=0.005,
		linecolor='silver',
		square=True,
		annot=True,
		fmt='.1f',
		annot_kws={'fontsize': 8},
	)
	kwargs = setup_kwargs(defaults, kwargs)
	fig, ax = create_figure(figsize=kwargs['figsize'])
	sns.heatmap(r, ax=ax, **filter_kwargs(sns.heatmap, kwargs))
	if title is not None:
		ax.set_title(
			label=title,
			y=kwargs['title_y'],
			fontsize=kwargs['title_fontsize'],
		)
	if xticklabels is not None:
		ax.set_xticklabels(xticklabels)
		ax.tick_params(
			axis='x',
			rotation=-90,
			labelsize=kwargs['tick_labelsize_x'],
		)
	if yticklabels is not None:
		ax.set_yticklabels(yticklabels)
		ax.tick_params(
			axis='y',
			rotation=0,
			labelsize=kwargs['tick_labelsize_y'],
		)
	if display:
		plt.show()
	else:
		plt.close()
	return fig, ax


def plot_latents_hist_full(
		z: np.ndarray,
		scales: List[int],
		display: bool = True,
		**kwargs, ):
	defaults = dict(
		bins_divive=128,
		figsize=None,
		figsize_x=3.25,
		figsize_y=2.75,
		layout='tight',
	)
	kwargs = setup_kwargs(defaults, kwargs)

	a = z.reshape((len(z), len(scales), -1))
	nrows, ncols = a.shape[1:]
	if kwargs['figsize'] is not None:
		figsize = kwargs['figsize']
	else:
		figsize = (
			kwargs['figsize_x'] * ncols,
			kwargs['figsize_y'] * nrows,
		)
	fig, axes = create_figure(
		nrows=nrows,
		ncols=ncols,
		sharey='row',
		figsize=figsize,
		layout=kwargs['layout'],
	)
	looper = itertools.product(
		range(nrows), range(ncols))
	for i, j in looper:
		x = a[:, i, j]
		sns.histplot(
			x,
			stat='percent',
			bins=len(a) // kwargs['bins_divive'],
			ax=axes[i, j],
		)
		msg = r"$\mu = $" + f"{x.mean():0.2f}, "
		msg += r"$\sigma = $" + f"{x.std():0.2f}\n"
		msg += f"minmax = ({x.min():0.2f}, {x.max():0.2f})\n"
		msg += f"skew = {sp_stats.skew(x):0.2f}"
		axes[i, j].set_title(msg)
	add_grid(axes)
	if display:
		plt.show()
	else:
		plt.close()
	return fig, axes


def plot_latents_hist(
		z: np.ndarray,
		scales: List[int],
		display: bool = True,
		**kwargs, ):
	defaults = dict(
		bins_divive=64,
		figsize=None,
		figsize_x=3.25,
		figsize_y=2.75,
		layout='tight',
	)
	kwargs = setup_kwargs(defaults, kwargs)

	nrows = 2
	ncols = int(np.ceil(len(scales) / nrows))
	if kwargs['figsize'] is not None:
		figsize = kwargs['figsize']
	else:
		figsize = (
			kwargs['figsize_x'] * ncols,
			kwargs['figsize_y'] * nrows,
		)
	fig, axes = create_figure(
		nrows=nrows,
		ncols=ncols,
		figsize=figsize,
		layout=kwargs['layout'],
	)
	a = z.reshape((len(z), len(scales), -1))
	for i in range(len(scales)):
		ax = axes.flat[i]
		x = a[:, i, :].ravel()
		sns.histplot(
			x,
			label=f"s = {scales[i]}",
			bins=len(a)//kwargs['bins_divive'],
			stat='percent',
			ax=ax,
		)
		msg = r"$\mu = $" + f"{x.mean():0.2f}, "
		msg += r"$\sigma = $" + f"{x.std():0.2f}\n"
		msg += f"minmax = ({x.min():0.2f}, {x.max():0.2f})\n"
		msg += f"skew = {sp_stats.skew(x):0.2f}"
		ax.set_ylabel('')
		ax.set_title(msg)
		ax.legend(loc='upper right')
	for i in range(2):
		axes[i, 0].set_ylabel('Proportion [%]')
	trim_axs(axes, len(scales))
	add_grid(axes)
	if display:
		plt.show()
	else:
		plt.close()
	return fig, axes
