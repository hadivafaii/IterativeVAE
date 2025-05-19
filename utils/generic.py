import re
import os
import json
import h5py
import time
import torch
import pickle
import joblib
import shutil
import random
import pathlib
import inspect
import logging
import argparse
import warnings
import operator
import functools
import itertools
import contextlib
import collections
import numpy as np
import pandas as pd
from torch import nn
from rich import print
from scipy import fft as sp_fft
from scipy import signal as sp_sig
from scipy import linalg as sp_lin
from scipy import stats as sp_stats
from scipy import ndimage as sp_img
from scipy import optimize as sp_optim
from scipy.spatial import distance as sp_dist
from scipy.spatial.transform import Rotation
from sklearn.preprocessing import Normalizer
from numpy.ma import masked_where as mwh
from torchvision.transforms.v2 import functional as F_vis
from torch.nn import functional as F
from prettytable import PrettyTable
from os.path import join as pjoin
from datetime import datetime
from tqdm import tqdm
from typing import *


def divide_integer(n, floats):
	allocations = [n * f for f in floats]
	ints = [int(a) for a in allocations]
	fractions = [a - int(a) for a in allocations]
	indices = sorted(
		range(len(fractions)),
		key=lambda e: fractions[e],
		reverse=True,
	)
	for i in range(n - sum(ints)):
		ints[indices[i]] += 1
	return ints


def divide_list(lst: list, n: int):
	k, m = divmod(len(lst), n)
	lst_divided = [
		lst[
			i * k + min(i, m):
			(i + 1) * k + min(i + 1, m)
		] for i in range(n)
	]
	return lst_divided


def invert_dict(input_dict: Dict[Any, dict]):
	inverted_dict = collections.defaultdict(dict)
	for outer_k, outer_dict in input_dict.items():
		for inner_k, inner_v in outer_dict.items():
			inverted_dict[inner_k][outer_k] = inner_v
	return dict(inverted_dict)


def shift_rescale(
	x: np.ndarray,
	loc: np.ndarray,
	scale: np.ndarray,
	fwd: bool = True, ):
	assert x.ndim == loc.ndim == scale.ndim
	return (x - loc) / scale if fwd else x * scale + loc


def interp(
	xi: Union[np.ndarray, torch.Tensor],
	xf: Union[np.ndarray, torch.Tensor],
	steps: int = 16, ):
	assert steps >= 2
	assert xi.shape == xf.shape
	shape = (steps, *xi.shape)
	if isinstance(xi, np.ndarray):
		x = np.empty(shape)
	elif isinstance(xi, torch.Tensor):
		x = torch.empty(shape)
	else:
		raise RuntimeError(type(xi))
	d = (xf - xi) / (steps - 1)
	for i in range(steps):
		x[i] = xi + i * d
	return x


def true_fn(s: str):  # used as argparse type
	return str(s).lower() == 'true'


def placeholder_fn(val, expected_type):  # used as argparse type
	return val if val == '__placeholder__' else expected_type(val)


def escape_parenthesis(fit_name: str):
	for s in fit_name.split('/'):
		print(s.replace('(', r'\(').replace(')', r'\)'))


def tensor_size(x: torch.Tensor, unit='gb'):
	size_bytes = x.element_size() * x.nelement()
	if unit.lower() == 'mb':
		return size_bytes / (1024 ** 2)
	elif unit.lower() == 'gb':
		return size_bytes / (1024 ** 3)
	else:
		msg = ' — '.join([
			f"invalid unit: {unit}",
			"choices: {'mb', 'gb'}",
		])
		raise ValueError(msg)


def int_from_str(s: str) -> int:
	matches = re.search(r"\d+", s)
	return int(matches.group())


def alphanum_sort_key(string: str):
	# \d+ matches one or more digits
	# [^\d]+ matches one or more non-digits
	chunks = re.findall(r"(\d+|\D+)", string)

	# Convert each chunk to either an
	# int or a str for natural sorting
	result = []
	for chunk in chunks:
		if chunk.isdigit():
			# Convert number chunks to int
			result.append(int(chunk))
		else:
			# Keep text chunks as strings
			result.append(str(chunk))
	return result


def tonp(x: Union[torch.Tensor, np.ndarray]):
	if isinstance(x, np.ndarray):
		return x
	elif isinstance(x, torch.Tensor):
		return x.data.cpu().numpy()
	elif isinstance(x, (list, tuple)):
		return np.asarray(x)
	else:
		raise ValueError(type(x).__name__)


def flat_cat(
		x_list: List[torch.Tensor],
		start_dim: int = 1,
		end_dim: int = -1,
		cat_dim: int = 1):
	x = [
		e.flatten(
			start_dim=start_dim,
			end_dim=end_dim,
		) for e in x_list
	]
	x = torch.cat(x, dim=cat_dim)
	return x


def flatten_np(
		x: np.ndarray,
		start_dim: int = 0,
		end_dim: int = -1, ):
	shape = x.shape
	if start_dim < 0:
		start_dim += len(shape)
	if end_dim < 0:
		end_dim += len(shape)
	prefix = shape[:start_dim]
	suffix = shape[end_dim+1:]
	middle = np.prod(shape[start_dim:end_dim+1])
	shape = (*prefix, middle, *suffix)
	return x.reshape(shape)


def flatten_arr(
		x: np.ndarray,
		ndim_end: int = 1,
		ndim_start: int = 0, ):
	shape = x.shape
	assert 0 <= ndim_end <= len(shape)
	assert 0 <= ndim_start <= len(shape)
	if ndim_end + ndim_start >= len(shape):
		return x

	shape_flat = shape[:ndim_start] + (-1,)
	for i, d in enumerate(shape):
		if i >= len(shape) - ndim_end:
			shape_flat += (d,)
	return x.reshape(shape_flat)


def avg(
		x: np.ndarray,
		ndim_end: int = 2,
		ndim_start: int = 0,
		fn: Callable = np.nanmean, ) -> np.ndarray:
	dims = range(ndim_start, x.ndim - ndim_end)
	dims = sorted(dims, reverse=True)
	for axis in dims:
		x = fn(x, axis=axis)
	return x


def cat_map(x: list, axis: int = 0):
	out = []
	for a in x:
		if len(a):
			out.append(np.concatenate(
				a, axis=axis))
		else:
			out.append(a)
	return out


def stack_map(x: list, axis: int = 0):
	out = []
	for a in x:
		if len(a):
			out.append(np.stack(
				a, axis=axis))
		else:
			out.append(a)
	return out


def get_tval(
		dof: int,
		ci: float = 0.95,
		two_sided: bool = True, ):
	if two_sided:
		ci = (1 + ci) / 2
	return sp_stats.t.ppf(ci, dof)


def contig_segments(mask: np.ndarray):
	censored = np.where(mask == 0)[0]
	looper = itertools.groupby(
		enumerate(censored),
		lambda t: t[0] - t[1],
	)
	segments = []
	for k, g in looper:
		s = map(operator.itemgetter(1), g)
		segments.append(list(s))
	return segments


def unique_idxs(
		obj: np.ndarray,
		filter_zero: bool = True, ):
	idxs = pd.DataFrame(obj.flat)
	idxs = idxs.groupby([0]).indices
	if filter_zero:
		idxs.pop(0, None)
	return idxs


def all_equal(iterable):
	g = itertools.groupby(iterable)
	return next(g, True) and not next(g, False)


def np_nans(shape: Union[int, Iterable[int]]):
	if isinstance(shape, np.ndarray):
		shape = shape.shape
	arr = np.empty(shape, dtype=float)
	arr[:] = np.nan
	return arr


def make_logger(
		name: str,
		path: str,
		level: int,
		module: str = None, ) -> logging.Logger:
	os.makedirs(path, exist_ok=True)
	logger = logging.getLogger(module)
	logger.setLevel(level)
	file = pjoin(path, f"{name}.log")
	file_handler = logging.FileHandler(file)
	formatter = logging.Formatter(
		'%(asctime)s : %(levelname)s : %(name)s : %(message)s')
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	return logger


def get_rng(
		x: int | np.random.Generator | random.Random = None,
		use_np: bool = True, ):
	"""
	Returns a random generator based on the input x.

	Parameters:
		- x: If an int, it's used as the seed. If None, a
			non-deterministic RNG is returned. Alternatively,
			x can be an instance of np.random.Generator or
			random.Random, in which case it is returned directly.
		- use_np: If True, use numpy's random generator;
			otherwise, use Python's built-in random.

	Returns:
		A random generator instance.
	"""
	if x is None:
		# No seed provided; return a new RNG with OS-provided entropy.
		return np.random.default_rng() if use_np else random.Random()
	elif isinstance(x, int):
		# An integer seed was provided.
		return np.random.default_rng(seed=x) if use_np else random.Random(x)
	elif isinstance(x, (np.random.Generator, random.Random)):
		# An RNG instance was provided.
		return x
	else:
		print("Warning: invalid random state. Returning default RNG.")
		return np.random.default_rng() if use_np else random.Random()


def add_home(path: str):
	if '/home/' not in path:
		return pjoin(os.environ['HOME'], path)
	return path


def setup_kwargs(defaults, kwargs):
	if not kwargs:
		return defaults
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v
	return kwargs


def get_default_params(f: Callable):
	params = inspect.signature(f).parameters
	return {
		k: p.default for
		k, p in params.items()
	}


def get_all_init_params(cls: Callable):
	init_params = {}
	classes_to_process = [cls]

	while classes_to_process:
		current_cls = classes_to_process.pop(0)
		for base_cls in current_cls.__bases__:
			if base_cls != object:
				classes_to_process.append(base_cls)
		sig = inspect.signature(current_cls.__init__)
		init_params.update(sig.parameters)

	return init_params


def filter_kwargs(
		fn: Callable,
		kw: dict = None, ):
	if not kw:
		return {}
	try:
		if isinstance(fn, type):  # class
			params = get_all_init_params(fn)
		elif callable(fn):  # function
			params = inspect.signature(fn).parameters
		else:
			raise ValueError(type(fn).__name__)
		return {
			k: v for k, v
			in kw.items()
			if k in params
		}
	except ValueError:
		return kw


def obj_attrs(
		obj: object,
		with_base: bool = True, ):
	# get params
	sig = inspect.signature(obj.__init__)
	params = dict(sig.parameters)
	if with_base:
		params.update(get_all_init_params(type(obj)))
	# get rid of self, args, kwargs
	vals = {
		k: getattr(obj, k) for
		k, p in params.items()
		if _param_checker(k, p, obj)
	}
	# get rid of functions
	vals = {
		k: v for k, v in vals.items()
		if not isinstance(v, Callable)
	}
	# remove directories
	vals = {
		k: v for k, v in vals.items()
		if '_dir' not in k
	}
	return vals


def save_obj(
		obj: Any,
		file_name: str,
		save_dir: str,
		mode: str = None,
		verbose: bool = True,
		**kwargs, ):
	_allowed_modes = [
		'npy', 'df',
		'pkl', 'joblib',
		'html', 'json', 'txt',
	]
	_ext = file_name.split('.')[-1]
	if _ext in _allowed_modes:
		mode = _ext
	else:
		if mode is None:
			msg = 'invalid file extension: '
			msg += f"{_ext}, mode: {mode}"
			raise RuntimeError(msg)
		else:
			file_name = f"{file_name}.{mode}"
	assert mode in _allowed_modes, \
		f"available modes:\n{_allowed_modes}"

	path = pjoin(save_dir, file_name)
	op_mode = 'w' if mode in ['html', 'json', 'txt'] else 'wb'
	with open(path, op_mode) as f:
		if mode == 'npy':
			np.save(f.name, obj, **kwargs)
		elif mode == 'df':
			pd.to_pickle(obj, f.name, **kwargs)
		elif mode == 'pkl':
			# noinspection PyTypeChecker
			pickle.dump(obj, f, **kwargs)
		elif mode == 'joblib':
			joblib.dump(obj, f, **kwargs)
		elif mode == 'html':
			f.write(obj)
		elif mode == 'json':
			# noinspection PyTypeChecker
			json.dump(obj, f, indent=4, **kwargs)
		elif mode == 'txt':
			for line in obj:
				f.write(line)
		else:
			raise RuntimeError(mode)
	if verbose:
		print(f"[PROGRESS] '{file_name}' saved at\n{save_dir}")
		return
	return path


def merge_dicts(
		dict_list: List[dict],
		verbose: bool = False, ) -> Dict[str, list]:
	merged = collections.defaultdict(list)
	dict_items = map(operator.methodcaller('items'), dict_list)
	iterable = itertools.chain.from_iterable(dict_items)
	kws = {
		'leave': False,
		'disable': not verbose,
		'desc': "...merging dicts",
	}
	for k, v in tqdm(iterable, **kws):
		merged[k].extend(v)
	return dict(merged)


def cumulative_mean(a: np.ndarray) -> np.ndarray:
	a = np.asarray(a)
	cumsum = np.cumsum(a)
	counts = np.arange(1, len(a) + 1)
	return cumsum / counts


def find_cumulative_convergence(
		arr: np.ndarray,
		window_size: int = 5,
		tol: float = 1e-5, ) -> int:
	if window_size < 1:
		raise ValueError("window_size must be at least 1.")
	if tol < 0:
		raise ValueError("tol must be non-negative.")

	cum_mean = cumulative_mean(arr)
	diffs = np.abs(np.diff(cum_mean))

	for i in range(len(diffs) - window_size + 1):
		window = diffs[i:i + window_size]
		if np.all(window < tol):
			return i + window_size
	return -1


def find_factors(n: int, m: int):
	factors = []
	factor = int(np.ceil(n**(1.0/m)))
	while n % factor != 0:
		factor = factor - 1
	factors.append(factor)
	if m > 1:
		factors = factors + find_factors(
			n / factor, m - 1)
	return factors


def find_last_contiguous_zeros(mask: np.ndarray, w: int):
	# mask = hist > 0.0
	m = mask.astype(bool)
	zero_count = 0
	for idx, val in enumerate(m[::-1]):
		if val == 0:
			zero_count += 1
		else:
			zero_count = 0

		if zero_count == w:
			return len(m) - (idx - w + 2)
	return 0


def find_critical_ids(mask: np.ndarray):
	# mask = hist > 0.0
	m = mask.astype(bool)

	first_zero = 0
	for i in range(1, len(m)):
		if m[i-1] and not m[i]:
			first_zero = i
			break

	last_zero = -1
	for i in range(len(m) - 2, -1, -1):
		if not m[i] and m[i+1]:
			last_zero = i
			break

	return first_zero, last_zero


def first_occurrence_inds(
		items: np.ndarray | torch.Tensor,
		num: int = 10, ):
	first_occurrences = {}
	first_indices = []

	for i, item in enumerate(tonp(items)):
		if item not in first_occurrences:
			first_occurrences[item] = i
			first_indices.append(i)
		if len(first_occurrences) == num:
			break
	return np.asarray(first_indices)


def next_power_of_two(x: int) -> int:
	if x <= 0:
		return 1
	return 2 ** (x - 1).bit_length()


def nearest_power_of_two(x: int) -> int:
	x = int(x)
	if x <= 0:
		return 1

	lower_power = 2 ** (x.bit_length() - 1)
	upper_power = 2 ** x.bit_length()

	diff_lower = x - lower_power
	diff_upper = upper_power - x

	if diff_lower <= diff_upper:
		return lower_power
	else:
		return upper_power


def base2(number: int):
	b = np.base_repr(number, base=2)
	if len(b) == 1:
		return 0, 0, int(b)
	elif len(b) == 2:
		j, k = b
		return 0, int(j), int(k)
	elif len(b) == 3:
		i, j, k = b
		return int(i), int(j), int(k)
	else:
		return b


def time_dff_string(start: str, stop: str):
	d, h, m, _ = time_difference(start, stop)
	delta_t = f"{h}h, {m}m"
	if d > 0:
		delta_t = f"{d}d, {delta_t}"
	return delta_t


def time_difference(
		start: str,
		stop: str,
		fmt: str = '%Y_%m_%d,%H:%M', ):
	start_datetime = datetime.strptime(start, fmt)
	stop_datetime = datetime.strptime(stop, fmt)
	diff = stop_datetime - start_datetime

	hours, remainder = divmod(diff.seconds, 3600)
	mins, seconds = divmod(remainder, 60)

	return diff.days, hours, mins, seconds


def now(include_hour_min: bool = False):
	fmt = "%Y_%m_%d"
	if include_hour_min:
		fmt += ",%H:%M"
	return datetime.now().strftime(fmt)


def _param_checker(k, p, obj):
	# 2nd cond gets rid of args, kwargs
	return k != 'self' and int(p.kind) == 1 and hasattr(obj, k)
