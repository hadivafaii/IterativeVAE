from .plotting import *
from IPython.display import HTML
from matplotlib.animation import FuncAnimation


def show_movie(
		x: np.ndarray,
		n_frames: int = -1,
		**kwargs, ):
	defaults = dict(
		figsize=(4.0, 3.0),
		cmap='Greys_r',
		vmin=None,
		vmax=None,
		interval=50,  # 50ms between frames
		blit=True,
	)
	kwargs = setup_kwargs(defaults, kwargs)

	fig, ax = plt.subplots(
		figsize=kwargs['figsize'])

	if n_frames == -1:
		n_frames = len(x)
	_x = tonp(x[:n_frames])

	if kwargs['vmin'] is None:
		kwargs['vmin'] = np.nanmin(_x)
	if kwargs['vmax'] is None:
		kwargs['vmax'] = np.nanmax(_x)

	# Show first frame and set colormap
	im = ax.imshow(
		_x[0],
		vmin=kwargs['vmin'],
		vmax=kwargs['vmax'],
		cmap=kwargs['cmap'],
	)
	plt.colorbar(im)
	remove_ticks(ax, False)

	def _update(i):
		im.set_array(_x[i])
		return [im]

	anim = FuncAnimation(
		fig=fig,
		func=_update,
		frames=len(_x),
		interval=kwargs['interval'],
		blit=kwargs['blit'],
	)

	# Convert animation to HTML5 video
	html_video = anim.to_jshtml()
	plt.close()
	return HTML(html_video)
