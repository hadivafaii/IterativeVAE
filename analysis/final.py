from base.utils_model import *
from figures.fighelper import get_palette_models
from figures.convergence import plot_convergence
from .linear import clf_score, untangle_score
import gc


def perform_analysis(
		results_dir: str,
		device: str | torch.device,
		fits: List[str] = None,
		kws_analysis: dict = None,
		kws_untangle: dict = None,
		attrs: Sequence[str] = None,
		attrs_tr: Sequence[str] = None,
		override_fits: bool = False,
		verbose: bool = True, ):

	attrs = attrs or [
		'seed', 'dataset', 'clamp_u',
		'type', 'latent_act', 'str_model',
		't_train', 'beta_outer', 'n_latents',
	]
	attrs_tr = attrs_tr or [
		'batch_size', 'epochs', 'lr',
	]

	kws_analysis = kws_analysis or dict(
		dl='vld',
		t_total=1000,
		n_data_batches=-1,
		data_batch_sz=1000,
		verbose=False,
	)
	kws_untangle = kws_untangle or dict(
		t_total=1000,
		save_freq=50,
		data_batch_sz=5000,
		verbose=False,
	)

	root = get_root_path()

	if fits is None:
		fits = sorted(
			os.listdir(root),
			key=alphanum_sort_key,
		)

	# the big for loop
	for name in tqdm(fits):
		save_name = f"{name}.npy"
		is_already_fit = os.path.isfile(pjoin(
			results_dir, save_name))
		if is_already_fit and not override_fits:
			continue

		# (1) load trainer
		try:
			tr, meta = load_model_lite(
				pjoin(root, name),
				device=device,
				shuffle=False,
				verbose=False,
				strict=True,
			)
		except StopIteration:
			print(f"missing: {name}")
			continue

		# (2) info
		info = {k: meta[k] for k in ['checkpoint', 'timestamp']}
		info.update({a: getattr(tr.model.cfg, a, None) for a in attrs})
		info['archi'] = tr.model.cfg.attr2archi()
		info.update({a: getattr(tr.cfg, a, None) for a in attrs_tr})
		info['n_params'] = sum([p.numel() for p in tr.parameters()])
		info['n_iters_train'] = tr.n_iters

		# (3) dataset-specific analyses
		if tr.model.cfg.dataset.startswith(('vH', 'Kyoto')):
			results = tr.analysis(**kws_analysis)
			# get map estimate r2
			state = tr.to(results['state_final'])
			state = tr.model.apply_act_fn(state)
			pred = tr.model.layer.phi(state)
			true = tr.dl_vld.dataset.tensors[0].flatten(start_dim=1)
			results['r2_map_final'] = compute_r2(
				true=true, pred=pred).mean().item()
			to_plot = results

		elif tr.model.cfg.dataset.endswith(('MNIST', 'SVHN')):
			results = clf_score(
				trainer=tr, **kws_untangle)
			del results['results_xtract']['trn']
			to_plot = results['results_xtract']['vld']

		elif tr.model.cfg.dataset.startswith('BALLS'):
			results = untangle_score(
				trainer=tr, **kws_untangle)
			del results['results_xtract']['trn']
			to_plot = results['results_xtract']['tst']

		elif tr.model.cfg.dataset == 'ImageNet32':
			continue

		else:
			raise NotImplementedError(tr.model.cfg.dataset)

		# (4) save
		save_obj(
			obj={'info': info, 'results': results},
			save_dir=results_dir,
			file_name=save_name,
			verbose=verbose,
		)

		# (5) plot?
		if verbose:
			nrows = 2
			items = None
			if not tr.model.cfg.dataset.startswith('vH'):
				nrows = 1
				items = ['mse', 'r2', 'du_norm']
			pal_models = get_palette_models()
			color = pal_models[tr.model.cfg.str_model]
			_ = plot_convergence(
				results=to_plot,
				nrows=nrows,
				items=items,
				color=color,
			)
			print('-' * 100)
			print('\n\n')

		# (6) clean up
		del tr
		torch.cuda.empty_cache()
		gc.collect()

	return


def get_root_path(base_path: str = 'Dropbox/chkpts'):
	current_dir = os.path.abspath(__file__)
	current_dir = os.path.dirname(current_dir)
	project_dir = os.path.dirname(current_dir)
	project_name = os.path.basename(project_dir)
	root = pjoin(base_path, project_name)
	root = add_home(root)
	return root
