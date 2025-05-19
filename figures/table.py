from utils.generic import *
from analysis.stats import mu_and_err


def table_mnist(
		df: pd.DataFrame,
		verbose: bool = True,
		**kwargs, ):

	table_elements = {}
	for model_family, mapping in _MODEL_MAPPING.items():
		# model family
		element_1 = f"{_PRE_C}\n{_SPACES}{model_family}\n{_POST}"
		# model latext codes
		element_2 = f" {_BS * 2} \n".join([
			f"{_SPACES}{model_latex_code}" for
			model_latex_code in mapping.values()
		])
		element_2 = f"{_PRE_L}\n{element_2}\n{_POST}"
		# table elements: info
		elements_info = [element_1, element_2]

		vals = collections.defaultdict(list)
		for str_model, model_latex_code in mapping.items():
			_df = df.loc[df['str_model'] == str_model]
			if not len(_df):
				continue

			# (1) reconstruction
			# MSE perdim
			data = np.asarray(_df['mse_perdim'])
			e_str_mse, *_ = mu_and_err(
				data=data,
				fmt='0.2f',
				**kwargs,
			)
			# R^2
			data = np.asarray(_df['r2'])
			e_str_r2, *_ = mu_and_err(
				data=data,
				fmt='0.3f',
				**kwargs,
			)
			e_str = ' & '.join([e_str_mse, e_str_r2])
			vals['reconstruction'].append(e_str)

			# (2) sparsity
			data = np.asarray(_df['%-zeros'])
			e_str, *_ = mu_and_err(
				data=data,
				fmt='0.2f',
				**kwargs,
			)
			vals['portion_zeros'].append(e_str)

			# (3) clf accuracy
			data = np.asarray(_df['accu'])
			e_str, *_ = mu_and_err(
				data=data,
				fmt='0.2f',
				**kwargs,
			)
			vals['clf_accuracy'].append(e_str)

			# (4) num params
			n_params = _df['n_params'].mean().item()
			n_params /= 1e6
			vals['num_params'].append(f"${n_params:0.2f}~M$")

		vals = {
			k: [f"{_SPACES}{e}" for e in v_list]
			for k, v_list in vals.items()
		}
		vals = {
			k: ' \\\ \n'.join(v_list)
			for k, v_list in vals.items()
		}

		elements_val = []
		for k, val_str in vals.items():
			if k == 'reconstruction':
				elem = f"{_PRE_X}\n{val_str}\n{_POST_X}"
			else:
				elem = f"{_PRE_C}\n{val_str}\n{_POST}"
			elements_val.append(elem)

		elements_final = elements_info + elements_val
		table = '\n&\n'.join(elements_final)
		table_elements[model_family] = table

	if verbose:
		for k, v in table_elements.items():
			print(f"### {k}\n" + '-' * 50)
			print(v)
			print('\n' + '-' * 50)
			print('\n\n')

	return table_elements


_BS = '\\'
_SPACES = ' ' * 4

_MODEL_MAPPING = {
	f"{_BS*2}".join(['iterative', 'VAE']): {
		'poisson': _BS + 'ipvae',
		'gaussian': _BS + 'igvae',
		'gaussian+relu': _BS + 'igreluvae',
	},
	f"{_BS*2}".join(['amortized', 'VAE']): {
		'poisson+amort': _BS + 'pvae',
		'gaussian+amort': _BS + 'gvae',
		'gaussian+relu+amort': _BS + 'greluvae',
	},
	# f"{_BS*2}".join(['Predictive', 'Coding', 'Net']): {
	'PCN': {
		'pc': 'PC \\cite{rao1999predictive}',
		'ipc': 'iPC  \\cite{salvatori2024ipc}',
		# 'dcpc': 'DCPC \\cite{sennesh2024ddpc}',
	},
}

_PRE_C = _BS + 'begin{tabular}{c}'
_PRE_L = _BS + 'begin{tabular}{l}'
_POST = _BS + 'end{tabular}'

_PRE_X = _BS + 'begin{tabularx}{\mycolwidth}{CC}'
_POST_X = _BS + 'end{tabularx}'
