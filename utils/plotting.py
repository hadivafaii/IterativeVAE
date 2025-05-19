from .generic import *
import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt
from IPython.display import display as display_ipy
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages
from matplotlib.colors import (
	to_rgb, rgb2hex, Colormap,
	ListedColormap, LinearSegmentedColormap,
)


def imshow(*args, **kwargs):
	args = (tonp(args[0]).squeeze(), *args[1:])
	return plt.imshow(*args, **kwargs)


def plot(*args, **kwargs):
	args = (tonp(args[0]).squeeze(), *args[1:])
	return plt.plot(*args, **kwargs)


def histplot(*args, **kwargs):
	kwargs.setdefault('element', 'step')
	kwargs.setdefault('fill', False)
	args = (tonp(args[0]), *args[1:])
	return sns.histplot(*args, **kwargs)


def make_ticks(
		span: Iterable[float],
		tick_spacing: int, ):
	ticks, ticklabels = zip(*[
		(x, (str(x))) for i, x
		in enumerate(span) if
		i % tick_spacing == 0
	])
	return ticks, ticklabels


def barplot_add_vals(
		axes,
		frac_x: float = 0.5,
		frac_y: float = 0.035,
		fontsize: float = 10,
		rotation: float = 0,
		min_h: float = 0.05,
		decimals: int = 2,
		ha: str = 'center',
		color: str = 'k', ):
	ymax = axes.get_ybound()[1]

	def _single(_ax):
		for p in _ax.patches:
			h = p.get_height()
			w = p.get_width()
			if h < min_h:
				continue
			_ax.text(
				x=p.get_x() + frac_x * w,
				y=p.get_y() + h - frac_y * ymax,
				s=np.round(h, decimals),
				fontsize=fontsize,
				rotation=rotation,
				color=color,
				ha=ha,
			)
	if isinstance(axes, np.ndarray):
		for idx, ax in np.ndenumerate(axes):
			_single(ax)
	else:
		_single(axes)
	return


# noinspection PyUnresolvedReferences
def show_hsv(figsize=(3, 3), num=2048):
	fig = plt.figure(figsize=figsize)
	cax = fig.add_axes((0, 0, 1, 1), projection='polar')
	cbar = matplotlib.colorbar.ColorbarBase(
		ax=cax,
		cmap=matplotlib.cm.get_cmap('hsv', num),
		norm=matplotlib.colors.Normalize(0, 2*np.pi),
		orientation='horizontal',
	)
	cax.get_children()[1].set_lw(0)
	cax.axis('off')
	plt.show()
	return fig, cax, cbar


# noinspection PyUnresolvedReferences
def cbar_only(
		cmap,
		vmin: float = 0,
		vmax: float = 1,
		vertical: bool = True,
		**kwargs, ):
	defaults = {
		'figsize': (0.4, 4) if
		vertical else (4, 0.35),
		'edgecolor': 'k',
		'linewidth': 1.3,
		'tick_pad': 2,
		'tick_length': 6,
		'tick_labelsize': 12,
		'tick_position': 'right'
		if vertical else 'bottom',
	}
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v

	fig, cax = create_figure(
		nrows=1,
		ncols=1,
		figsize=kwargs['figsize'],
		constrained_layout=False,
		tight_layout=False,
	)
	cbar = matplotlib.colorbar.ColorbarBase(
		ax=cax,
		cmap=matplotlib.cm.get_cmap(cmap)
		if isinstance(cmap, str) else cmap,
		norm=matplotlib.colors.Normalize(vmin, vmax),
		orientation='vertical' if vertical else 'horizontal',
	)

	cbar.outline.set_edgecolor(kwargs['edgecolor'])
	cbar.outline.set_linewidth(kwargs['linewidth'])

	cax.tick_params(
		axis='y' if vertical else 'x',
		pad=kwargs['tick_pad'],
		length=kwargs['tick_length'],
		labelsize=kwargs['tick_labelsize'],
		color=kwargs['edgecolor'],
		width=kwargs['linewidth'],
	)
	if vertical:
		cax.yaxis.set_ticks_position(kwargs['tick_position'])
	else:
		cax.xaxis.set_ticks_position(kwargs['tick_position'])
	plt.close()
	return fig, cax, cbar


def add_jitter(
		x: np.ndarray,
		sigma: float = 0.01,
		shift_mean: bool = True, ):
	jit = get_rng().normal(scale=sigma, size=len(x))
	if shift_mean:
		jit -= jit.mean()
	return x + jit


def _iter_ax(axes):
	if not isinstance(axes, Iterable):
		return [axes]
	elif isinstance(axes, np.ndarray):
		return axes.flat


def ax_square(axes):
	for ax in _iter_ax(axes):
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		delta_x = max(xlim) - min(xlim)
		delta_y = max(ylim) - min(ylim)
		aspect = delta_x / delta_y
		ax.set_aspect(aspect, adjustable='box')
	return axes


def move_legend(
		ax: plt.Axes,
		bbox: Tuple[float, float] = None, ):
	leg = ax.get_legend()
	if leg is None:
		return ax
	if bbox is None:
		leg.remove()
	else:
		leg.set_bbox_to_anchor(bbox)
	return ax


def trim_axs(axes, n):
	axs = axes.flat
	for ax in axs[n:]:
		ax.remove()
	return axs[:n]


def add_grid(axes, **kwargs):
	for ax in _iter_ax(axes):
		ax.grid(**kwargs)
	return axes


def add_legend(axes, **kwargs):
	for ax in _iter_ax(axes):
		ax.legend(**kwargs)
	return axes


def add_ax_inset(
		ax: plt.Axes,
		data: np.ndarray,
		bounds: Tuple[float, float, float, float],
		kind: str = 'imshow',
		aspect_eq: bool = True,
		**kwargs, ):
	axins = ax.inset_axes(bounds)
	if kind == 'imshow':
		x2p = np.ma.masked_where(data[0] == 0, data[0])
		axins.imshow(x2p, cmap='Greys_r')
		x2p = np.ma.masked_where(data[1] < kwargs['vmin'], data[1])
		axins.imshow(x2p, **kwargs)
	elif kind == 'kde':
		sns.kdeplot(data=data, ax=axins, **kwargs)
	else:
		raise NotImplementedError
	if aspect_eq:
		axins.set_aspect('equal', adjustable='box')
	return axins


def remove_ticks(axes, full=True):
	for ax in _iter_ax(axes):
		ax.set_xticks([])
		ax.set_yticks([])
		if full:
			try:
				_ = list(map(
					lambda z: z.set_visible(False),
					ax.spines.values()
				))
			except AttributeError:
				continue
	return


def save_marker_as_pdf(
		marker: str,
		save_dir: str,
		file_name: str = None,
		face_color: str = 'C0',
		edge_color: str = 'dimgrey',
		face_alpha: float = 0.4,
		edge_alpha: float = 1.0,
		markersize: float = 100,
		edge_linewidth: float = 11.0, ):
	# Create a figure with transparent background
	fig, ax = plt.subplots(figsize=(2.1, 2.1))
	fig.patch.set_alpha(0)
	ax.set_aspect('equal')

	# Set up a clean, minimal plot area
	ax.set_xlim(-0.5, 0.5)
	ax.set_ylim(-0.5, 0.5)
	ax.axis('off')

	# Plot the edge (border) first
	ax.plot(
		0, 0,
		marker=marker,
		markerfacecolor=face_color,
		markeredgecolor='none',
		markersize=markersize * 0.98,
		alpha=face_alpha,
		linestyle='none'
	)
	ax.plot(
		0, 0,
		marker=marker,
		markerfacecolor='none',
		markeredgecolor=edge_color,
		markersize=markersize,
		markeredgewidth=edge_linewidth,
		alpha=edge_alpha,
		linestyle='none'
	)

	if file_name is None:
		file_name = f"marker_{marker}"
	file_name = f"{file_name}.pdf"
	fig.savefig(
		pjoin(save_dir, file_name),
		format='pdf',
		bbox_inches='tight',
		pad_inches=0.1,
		transparent=True
	)
	plt.show(fig)
	return


def py_color_to_latex(cmap: str, name_str: str):
	cmap = sns.color_palette(cmap)
	colors = [rgb2hex(c) for c in cmap]

	commands = []
	for i, c in enumerate(colors):
		cmd = (
			'\\definecolor{'
			+ f"{name_str}_{i}"
			+ '}{HTML}{'
			+ c.lstrip('#').upper()
			+ '}'
		)
		commands.append(cmd)
	commands = '\n'.join(commands)
	print(commands)
	return


# noinspection PyTypeChecker
def make_cmap(
		hex_colors: List[str],
		name: str = 'custom_cmap',
		n_colors: int = 256,
		show: bool = True, ):
	colors = [to_rgb(c) for c in hex_colors]
	color_ramp = LinearSegmentedColormap.from_list(
		name=name, colors=colors, N=n_colors)
	if show:
		display_cmap(color_ramp, len(hex_colors))
	return color_ramp


def get_cubehelix_palette(n_colors: int, **kwargs):
	defaults = dict(
		start=2.5,
		rot=0.3,
		gamma=1.0,
		hue=0.9,
		light=0.85,
		dark=0.15,
		reverse=False,
		as_cmap=False,
	)
	kwargs = setup_kwargs(defaults, kwargs)
	pal = sns.cubehelix_palette(
		n_colors=n_colors, **kwargs)
	return pal


# noinspection PyTypeChecker
def get_rand_cmap(num: int, rng=None):
	rng = rng if rng else get_rng()
	colors = (
		rng.choice(256, size=num) / 256,
		rng.choice(256, size=num) / 256,
		rng.choice(256, size=num) / 256,
	)
	colors = [
		rgb2hex(c) for c in
		list(zip(*colors))
	]
	cmap = make_cmap(
		hex_colors=colors,
		name='random',
		n_colors=num,
		show=False,
	)
	return cmap


def get_hm_cmap(
		colors: List[str] = None,
		return_clist: bool = False, ):
	colors = colors if colors else [
		'#365c7f', '#09f6e1', '#7ac837',
		'#fbf504', '#f0640f', '#f50a10',
	]
	heatmap = make_cmap(
		hex_colors=colors,
		name='heatmap',
		show=False,
	)
	if return_clist:
		return heatmap, colors
	else:
		return heatmap


# noinspection PyTypeChecker
def get_periodic_palette(n_colors: int = 180):
	base_colors = sns.color_palette("hls", n_colors)
	periodic_cmap = ListedColormap(base_colors)
	return periodic_cmap, base_colors


def rd_or_bu(reverse: bool = False):
	colors = [
		'#ed3800', '#ff6100', '#ffc8a7',
		'#FFFFFF',
		'#94d6ff', '#009EFF', '#0762df',
	]
	if reverse:
		colors = colors[::-1]
	cmap = make_cmap(
		hex_colors=colors,
		name='RdOrBu',
		n_colors=512,
		show=False,
	)
	return cmap


def display_cmap(cmap: Union[str, Colormap], n: int = 256):
	fig, ax = create_figure(1, 1, (13.5, 3), layout='tight')
	ax.imshow(
		np.array([list(np.arange(0, n, 0.11 * n / 4))]),
		interpolation='nearest',
		origin='lower',
		cmap=cmap,
	)
	remove_ticks(ax)
	plt.show()
	return


def cmap2pal(cmap: Colormap, n_colors: int, start: float):
	x = np.linspace(start % 1, 1.0, n_colors)
	x = [cmap(abs(i - 1e-10)) for i in x]
	return sns.color_palette(x)


# noinspection PyTypeChecker
def complement_color(r, g, b):
	def _hilo(i, j, k):
		if k < j:
			j, k = k, j
		if j < i:
			i, j = j, i
		if k < j:
			j, k = k, j
		return i + k

	h = _hilo(r, g, b)
	return rgb2hex(tuple(h - u for u in (r, g, b)))


def fonts_html():
	import matplotlib.font_manager

	def _mk_html(fontname):
		html = "<p>{font}: <span style='font-family:{font}; "
		html += "font-size: 24px;'>{font}  01234 </p>"
		return html.format(font=fontname)

	code = sorted(set([f.name for f in matplotlib.font_manager.fontManager.ttflist]))
	code = [_mk_html(font) for font in code]
	code = "\n".join(code)
	return "<div style='column-count: 2;'>{}</div>".format(code)


# noinspection PyTypeChecker
def set_style(
		context: str = 'notebook',
		style: str = 'ticks',
		palette: str = None,
		font: str = 'sans-serif', ):
	sns.set_theme(
		context=context,
		style=style,
		palette=palette,
		font=font,
	)
	matplotlib.rcParams['grid.linestyle'] = ':'
	matplotlib.rcParams['figure.figsize'] = (3.0, 2.0)
	matplotlib.rcParams['image.interpolation'] = 'none'
	matplotlib.rcParams['font.family'] = font
	return


def create_figure(
		nrows: int = 1,
		ncols: int = 1,
		figsize: Tuple[float, float] = None,
		sharex: Literal['none', 'all', 'row', 'col'] = 'none',
		sharey: Literal['none', 'all', 'row', 'col'] = 'none',
		layout: str = None,
		wspace: float = None,
		hspace: float = None,
		width_ratios: List[float] = None,
		height_ratios: List[float] = None,
		reshape: bool = False,
		style: str = 'ticks',
		dpi: float = None,
		cnst: bool = True,
		**kwargs, ):
	"""
	:param nrows:
	:param ncols:
	:param figsize:
	:param layout: {'constrained', 'compressed', 'tight', None}
	:param sharex: {'none', 'all', 'row', 'col'} or bool
	:param sharey: {'none', 'all', 'row', 'col'} or bool
	:param style: {'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'}
	:param wspace:
	:param hspace:
	:param width_ratios:
	:param height_ratios:
	:param reshape:
	:param dpi:
	:param cnst: shortcut to turn on layout='constrained'
	:param kwargs:
	:return: fig, axes
	"""
	set_style(style=style)
	figsize = figsize or [
		mult * default_size for mult, default_size in
		zip((ncols, nrows), plt.rcParams.get('figure.figsize'))
	]
	dpi = dpi if dpi else plt.rcParams.get('figure.dpi')
	layout = 'constrained' if cnst else layout

	fig, axes = plt.subplots(
		nrows=nrows,
		ncols=ncols,
		sharex=sharex,
		sharey=sharey,
		layout=layout,
		figsize=figsize,
		gridspec_kw={
			'wspace': wspace,
			'hspace': hspace,
			'width_ratios': width_ratios,
			'height_ratios': height_ratios},
		dpi=dpi,
		**kwargs,
	)
	if reshape:
		axes = np.array(axes).reshape((nrows, ncols))
	return fig, axes


def save_fig(
		fname: str,
		save_dir: str,
		fig: plt.Figure | List[plt.Figure],
		sup: plt.Text | List[plt.Text] = None,
		display: bool = False,
		**kwargs, ):
	defaults = {
		'dpi': 300,
		'bbox_inches': 'tight',
		'transparent': True,
	}
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v
	os.makedirs(save_dir, exist_ok=True)
	save_file = pjoin(save_dir, fname)

	if isinstance(fig, plt.Figure):
		fig.savefig(
			fname=save_file,
			bbox_extra_artists=sup,
			**kwargs,
		)
	else:
		sup = sup or [None] * len(fig)
		assert fname.split('.')[-1] == 'pdf'
		assert len(fig) == len(sup) > 1
		with PdfPages(save_file) as pages:
			for f, s in zip(fig, sup):
				if f is None:
					continue
				canvas = FigureCanvasPdf(f)
				if s is not None:
					canvas.print_figure(
						filename=pages,
						bbox_extra_artists=[s],
						**kwargs,
					)
				else:
					canvas.print_figure(
						filename=pages,
						**kwargs,
					)
	if display:
		if isinstance(fig, list):
			for f in fig:
				display_ipy(f)
		else:
			display_ipy(fig)
	if isinstance(fig, list):
		for f in fig:
			plt.close(f)
	else:
		plt.close(fig)
	return save_file
