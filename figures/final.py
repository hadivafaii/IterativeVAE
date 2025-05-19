from .fighelper import *


def fig_ratedist_box(
		df_ratedist: pd.DataFrame,
		metric: str = 'distance',
		labels: bool = False,
		**kwargs, ):
	defaults = dict(
		figsize=(2.0, 3.7),
		jitter_sigma=0.08,
		scatter_alpha=0.7,
		border_alpha=1.0,
		scatter_size=5,
		showmeans=False,
		yticksize=10,
	)
	kwargs = setup_kwargs(defaults, kwargs)
	model_order = [
		'gaussian+amort',
		'gaussian',
		'gaussian+relu+amort',
		'gaussian+relu',
		'poisson+amort',
		'poisson',
		'lca',
	]
	pal_models = get_palette_models()
	fig, ax = create_figure(1, 1, kwargs['figsize'])

	# (1) scatter plot, jittered
	for i, model in enumerate(model_order):
		_df = df_ratedist[df_ratedist['str_model'] == model]
		if len(_df) > 0:
			x_pos = add_jitter(
				x=np.full(len(_df), i),
				sigma=kwargs['jitter_sigma'],
			)
			ax.scatter(
				x_pos,
				_df[metric],
				facecolors='none',
				edgecolors=pal_models[model],
				alpha=kwargs['scatter_alpha'],
				s=kwargs['scatter_size'],
				linewidths=0.5,
				zorder=3,
			)

	# (2) boxplot
	line_props = dict(
		linestyle='-', linewidth=1,
		alpha=kwargs['border_alpha'],
	)
	box_props = {
		'boxprops': {**line_props, 'facecolor': 'none'},
		'whiskerprops': line_props.copy(),
		'capprops': line_props.copy(),
		'medianprops': {**line_props, 'alpha': 1.0},
		'meanprops': dict(marker='o', markersize=4, markeredgewidth=0),
		'showfliers': False,
		'capwidths': 0.5,
		'zorder': 2
	}
	for i, model in enumerate(model_order):
		_df = df_ratedist[df_ratedist['str_model'] == model]
		if len(_df) > 0:
			color = pal_models[model]
			current_props = {
				k: {**v, 'color': color} for k, v in box_props.items() if
				k in ['boxprops', 'whiskerprops', 'capprops', 'medianprops']
			}
			current_props['meanprops'] = {
				**box_props['meanprops'],
				'markerfacecolor': color,
				'markeredgecolor': color,
			}
			for k in ['showfliers', 'capwidths', 'zorder']:
				current_props[k] = box_props[k]

			ax.boxplot(
				_df[metric],
				positions=[i],
				widths=0.80,
				patch_artist=True,
				showmeans=kwargs['showmeans'],
				**current_props
			)
	ax.tick_params(axis='y', labelsize=kwargs['yticksize'])
	ax.axhline(0, ls='--', color='silver')
	ax.set(xlabel='')
	add_grid(ax)

	if labels:
		ax.set_xticks(range(len(model_order)))
		ax.set_xticklabels(model_order, rotation=-90)
		lbl = r"$d = \sqrt{(1 - R2)^2 + (1-PZ)^2}$"
		ax.set(ylabel=lbl)
	else:
		ax.set_xticklabels([])

	plt.show()
	return fig


def fig_ratedist_scatter(
		df_ratedist: pd.DataFrame,
		recon_metric: str = 'r2',
		labels: bool = False,
		**kwargs, ):
	defaults = dict(
		figsize=(6, 3.7),
		markersize=10,
		face_alpha=0.4,
	)
	kwargs = setup_kwargs(defaults, kwargs)

	df = df_ratedist.loc[
		~df_ratedist['latent_act'].isin([
			'softplus', 'sigmoid'])
	]
	pal_models = get_palette_models()
	fig, ax = create_figure(1, 1, kwargs['figsize'])

	legend_elements = []
	for model in df['str_model'].unique():
		model_color = pal_models[model]

		for seq_len in [1, 8, 16, 32, -1]:
			subset = df[
				(df['str_model'] == model) &
				(df['seq_len'] == seq_len)
			]
			if len(subset) == 0:
				continue

			marker, fillstyle = get_marker_style(seq_len)

			if fillstyle == 'full':
				ax.plot(
					subset['%-zeros'],
					subset[recon_metric],
					marker=marker,
					markeredgecolor=model_color,
					markerfacecolor='none',
					markersize=kwargs['markersize'],
					alpha=1.0,
					linestyle='none'
				)
				ax.plot(
					subset['%-zeros'],
					subset[recon_metric],
					marker=marker,
					markeredgecolor='none',
					markerfacecolor=model_color,
					markersize=kwargs['markersize'] - 0.5,
					alpha=kwargs['face_alpha'],
					linestyle='none'
				)
			else:
				ax.plot(
					subset['%-zeros'],
					subset[recon_metric],
					marker=marker,
					fillstyle=fillstyle,
					color=model_color,
					markersize=kwargs['markersize'] - 0.5,
					alpha=1.0,
					linestyle='none'
				)

			if fillstyle == 'full':
				legend_element = matplotlib.lines.Line2D(
					[0], [0],
					marker=marker,
					markerfacecolor=model_color,
					markerfacecoloralt='none' if fillstyle == 'none' else None,
					markeredgecolor=model_color,
					markersize=8,
					linestyle='none',
					alpha=0.8,
					label=f"{model}, seq_len={seq_len}"
				)
			else:
				legend_element = matplotlib.lines.Line2D(
					[0], [0],
					marker=marker,
					markerfacecolor='none',
					markeredgecolor=model_color,
					markersize=8,
					linestyle='none',
					label=f"{model}, seq_len={seq_len}"
				)

			legend_elements.append(legend_element)

	ax.axvline(0, color='silver', ls='--', zorder=0)
	ax.axvline(1, color='silver', ls='--', zorder=0)
	ax.axhline(0, color='silver', ls='--', zorder=0)
	ax.axhline(1, color='silver', ls='--', zorder=0)

	if labels:
		ax.legend(handles=legend_elements)
		move_legend(ax, (1.02, 1.0))
		ax.set_xlabel('Sparsity [portion zeros]')
		ax.set_ylabel('Reconstruction [R^2]')
	else:
		ax.set(xlabel='', ylabel='')

	ax.grid()
	plt.show()

	return fig


def fig_convergence(
		df: pd.DataFrame,
		lws: dict = None,
		colors: dict = None,
		legend: bool = False,
		display: bool = True, ):
	lws = lws or {
		'r2': 2.5,
		'%-zeros': 2.5,
		'du_norm': 1.5,
	}
	colors = colors or {
		'r2': sns.color_palette('hls', 16)[10],
		'%-zeros': 'C2',  # sns.color_palette('hls', 16)[0],
		'du_norm': '#ED1C24',
	}
	# get selected data
	xs = sorted(df['time'].unique() + 1)
	models = ['poisson', 'gaussian+relu', 'gaussian']

	fig, axes = create_figure(1, 3, (8, 2.5), sharex='all', sharey='all')

	# Keep track of the twin axes
	right_axes = []

	# First collect all du_norm values to determine consistent y-axis limits
	all_du_norm_values = []
	for model in models:
		_df = df.loc[df['str_model'] == model]
		all_du_norm_values.extend(_df['du_norm'].values)

	# Calculate right y-axis limits
	du_norm_min = min(all_du_norm_values)
	du_norm_max = max(all_du_norm_values)

	if du_norm_min <= 0:
		positive_values = [v for v in all_du_norm_values if v > 0]
		if positive_values:
			du_norm_min = min(positive_values) * 0.9
		else:
			du_norm_min = 0.001
	else:
		du_norm_min *= 0.7
	du_norm_max *= 1.05

	for i, model in enumerate(models):
		_df = df.loc[df['str_model'] == model]
		ax = axes[i]

		# Create a twin axis and save reference
		ax_right = ax.twinx()
		right_axes.append(ax_right)

		# New plot on right y-axis
		ax_right.plot(
			xs, _df['du_norm'].values,
			color=colors['du_norm'],
			lw=lws['du_norm'],
			ls='--',
			label='du_norm',
			zorder=1,
		)

		ax_right.set(ylim=(du_norm_min, du_norm_max))
		if i < 2:
			ax_right.set(yticklabels=[])

		ax.plot(
			xs, _df['%-zeros'].values,
			color=colors['%-zeros'],
			lw=lws['%-zeros'],
			label='%-zeros',
			zorder=3,
		)
		ax.plot(
			xs, _df['r2'].values,
			color=colors['r2'],
			lw=lws['r2'],
			label='r2',
			zorder=4,
		)

		# Set scales and limits for left y-axis
		ylim_max = max(
			df['r2'].max(),
			df['%-zeros'].max(),
		) * 1.04
		ax.set(xscale='log', ylim=(-0.05, ylim_max))

		# Add model name
		if legend:
			ax.text(0.55, 0.3, model, transform=ax.transAxes, fontsize=8, zorder=5)
	if legend:
		lines_left, labels_left = axes[0].get_legend_handles_labels()
		lines_right, labels_right = right_axes[0].get_legend_handles_labels()
		lines = lines_left + lines_right
		labels = labels_left + labels_right
		fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3)

	add_grid(axes)

	if display:
		plt.show()
	else:
		plt.close()
	return fig
