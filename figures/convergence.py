from utils.plotting import *
from .imgs import plot_weights


def plot_row_or_col(
		x: np.ndarray,
		row: bool = True,
		display: bool = False,
		**kwargs, ):
	defaults = dict(
		cmap='Greys',
		method='none',
		vmin=0.05,
		vmax=1.0,
		dpi=80,
	)
	kwargs = setup_kwargs(defaults, kwargs)
	fig, ax = plot_weights(
		w=x,
		nrows=1 if row else len(x),
		display=display,
		**kwargs,
	)
	return fig, ax


def convergence_subplot(
		ax,
		data,
		label,
		legend_title='',
		intvl=None,
		legend=True,
		annot=False,
		xscale='log',
		yscale='log',
		**kwargs, ):
	defaults = {
		'figsize_x': 5.0,
		'figsize_y': 3.0,
		'legend_fontsize': 10,
		'color': 'C0',
		'marker': '.',
		'markersize': 6,
		'lw': 1,
		'zorder': None,
	}
	kwargs = setup_kwargs(defaults, kwargs)
	_intvl = intvl or range(0, len(data))
	_intvl = range(
		_intvl.start,
		min(_intvl.stop, len(data)),
		_intvl.step,
	)
	y = data[_intvl]

	if _intvl.start == 0:
		xs = [i + 1 for i in _intvl]
	else:
		xs = list(_intvl)

	ax.plot(
		xs, y,
		label=label,
		color=kwargs['color'],
		marker=kwargs['marker'],
		markersize=kwargs['markersize'],
		zorder=kwargs['zorder'],
		lw=kwargs['lw'],
	)

	if annot:
		fmt = '0.2g' if y[-1] > 1000 else '0.2f'
		label = r"$\lim_{t \rightarrow \infty}: $"
		label += f" {y[-1]:{fmt}}"
		ax.axhline(y[-1], color='dimgrey', ls='--')
		ax.annotate(
			text=label,
			xy=(0.02, y[-1]),
			xytext=(0, -14 if y[-1] > 1000 else 13),
			xycoords=('axes fraction', 'data'),
			textcoords='offset points',
			color='dimgrey',
			fontsize=13,
		)
	ax.set(xscale=xscale, yscale=yscale)
	if legend:
		ax.legend(
			title=legend_title,
			fontsize=kwargs['legend_fontsize'],
		)
	# ax.grid()
	return ax


def plot_convergence(
		results: dict,
		nrows: int = 2,
		items: List[str] = None,
		intvl: range = None,
		display: bool = True,
		**kwargs, ):
	defaults = {
		'figsize_x': 5.0,
		'figsize_y': 3.0,
		'legend_fontsize': 13,
		'color': 'C0',
		'marker': '.',
		'markersize': 6,
		'lw': 1,
	}
	kwargs = setup_kwargs(defaults, kwargs)

	items = items or [
		'mse', 'kl', 'r2', # 'nelbo',
		'du_norm', 'lifetime', '%-zeros',
	]
	ncols = int(np.ceil(len(items) / nrows))
	figsize = (
		kwargs['figsize_x'] * ncols,
		kwargs['figsize_y'] * nrows,
	)
	fig, axes = create_figure(
		nrows=nrows,
		ncols=ncols,
		figsize=figsize,
		sharex='col',
	)
	for ax, key in zip(axes.flat, items):
		kws = dict(
			ax=ax,
			data=results[key],
			label=key,
			intvl=intvl,
			kwargs=kwargs,
			xscale='log',
		)
		if key in ['lifetime', 'population', '%-zeros', 'r2']:
			kws['yscale'] = 'linear'
			kws['max_good'] = True
			_subplot(**kws)
		else:
			kws['yscale'] = 'log'
			_subplot(**kws)
	trim_axs(axes, len(items))

	if display:
		plt.show()
	else:
		plt.close()
	return fig, axes


def _subplot(
		ax,
		data,
		label,
		intvl,
		kwargs,
		xscale='log',
		yscale='log',
		max_good=False, ):
	_intvl = intvl or range(0, len(data))
	_intvl = range(
		_intvl.start,
		min(_intvl.stop, len(data)),
		_intvl.step,
	)
	y = data[_intvl]
	if np.isfinite(y).sum() == 0:
		return ax

	if max_good:
		best_i = np.nanargmax(y)
	else:
		best_i = np.nanargmin(y)

	if _intvl.start == 0:
		xs = [i + 1 for i in _intvl]
		shifted = True
	else:
		xs = list(_intvl)
		shifted = False

	ax.plot(
		xs, y,
		label=label,
		color=kwargs['color'],
		marker=kwargs['marker'],
		markersize=kwargs['markersize'],
		lw=kwargs['lw'],
	)
	i = best_i + 1 if shifted else best_i
	label = ' '.join([
		f"{'max' if max_good else 'min'}",
		f"(i = {best_i + 1}): {y[best_i]:0.2f}"
	])
	ax.axvline(i, color='g', ls='--', label=label)
	fmt = '0.2g' if y[-1] > 1000 else '0.2f'
	label = f"final: {y[-1]:{fmt}}"
	ax.axhline(y[-1], color='r', ls='--', label=label)

	ax.set(xscale=xscale, yscale=yscale)
	ax.legend(fontsize=kwargs['legend_fontsize'])
	ax.grid()
	return ax
