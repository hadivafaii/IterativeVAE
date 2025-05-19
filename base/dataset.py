from .utils_model import *
from PIL import ImageOps
import torchvision


def make_dataloader(
		dataset: str,
		device: torch.device,
		load_dir: str = 'Datasets',
		batch_size: int = 256,
		**kwargs, ):
	# datasets
	trn, vld, _ = make_dataset(
		dataset=dataset,
		device=device,
		load_dir=load_dir,
		**kwargs,
	)
	# dataloaders
	defaults = dict(
		batch_size=batch_size,
		drop_last=False,
		shuffle=False,
	)
	kwargs = filter_kwargs(
		fn=torch.utils.data.DataLoader,
		kw=setup_kwargs(defaults, kwargs),
	)
	dl_trn = torch.utils.data.DataLoader(trn, **kwargs)
	dl_vld = torch.utils.data.DataLoader(vld, **kwargs)
	return dl_trn, dl_vld


def make_dataset(
		dataset: str,
		device: torch.device = 'cpu',
		load_dir: str = 'Datasets',
		**kwargs, ):

	tst = None
	load_dir = add_home(load_dir)

	if any(s in dataset for s in ['vH', 'Kyoto']):
		defaults = dict(
			shift_rescale=True,
			vld_portion=0.2,
			n_blocks=10,
		)
		kwargs = setup_kwargs(defaults, kwargs)

		match dataset.split('-')[1]:
			case 'col': file_name = 'patches_color.npy'
			case 'wht': file_name = 'processed.npy'
			case 'bw': file_name = 'patches.npy'
			case _: file_name = 'patches.npy'

		data = pjoin(
			load_dir,
			'DOVES' if 'vH' in dataset else 'Kyoto',
			dataset.split('-')[0],
			file_name,
		)
		data = np.load(data)

		if kwargs['shift_rescale']:
			mu = np.nanmean(data)
			sd = np.nanstd(data)
			data = (data - mu) / sd

		trn_inds, vld_inds = split_data(
			n_samples=len(data),
			n_blocks=kwargs['n_blocks'],
			vld_portion=kwargs['vld_portion'],
		)
		trn, vld = data[trn_inds], data[vld_inds]
		trn, vld = map(
			lambda a: torch.tensor(
				data=a,
				device=device,
				dtype=torch.float),
			[trn, vld],
		)
		trn = torch.utils.data.TensorDataset(trn)
		vld = torch.utils.data.TensorDataset(vld)

	elif dataset == 'CIFAR16':
		path = pjoin(load_dir, 'CIFAR10', 'xtract16')
		trn = np.load(pjoin(path, 'trn', 'processed.npy'))
		vld = np.load(pjoin(path, 'vld', 'processed.npy'))

		trn, vld = map(_to_device_fun(device), [trn, vld])
		trn = torch.utils.data.TensorDataset(trn)
		vld = torch.utils.data.TensorDataset(vld)

	elif dataset in ['MNIST', 'FashionMNIST']:
		path = pjoin(load_dir, dataset, 'processed')
		if os.path.isfile(pjoin(path, 'x_trn.npy')):
			trn, vld, tst = _load(path, device)
		else:
			# get transforms
			transform = torchvision.transforms.Compose([
				torchvision.transforms.ToTensor(),
				ToDevice(device=device),
			])
			# make dataset
			kws = dict(root=load_dir, transform=transform)
			dataset = getattr(torchvision.datasets, dataset)
			trn = dataset(train=True, **kws)
			vld = dataset(train=False, **kws)
			# process and save
			trn, vld, _ = _process_and_save(
				trn=trn, vld=vld, save_dir=path)

	elif dataset == 'EMNIST':
		path = pjoin(load_dir, 'EMNIST', 'processed')
		if os.path.isfile(pjoin(path, 'x_trn.npy')):
			trn, vld, tst = _load(path, device)
		else:
			# get transforms
			transform = torchvision.transforms.Compose([
				torchvision.transforms.ToTensor(),
				ToDevice(device=device),
			])
			# make dataset
			kws = dict(root=load_dir, transform=transform, split='letters')
			trn = torchvision.datasets.EMNIST(train=True, **kws)
			vld = torchvision.datasets.EMNIST(train=False, **kws)
			# process and save
			trn, vld, _ = _process_and_save(
				trn=trn,
				vld=vld,
				save_dir=path,
				transpose=True,
			)

	elif dataset == 'Omniglot':
		path = pjoin(load_dir, 'omniglot-py', 'processed')
		if os.path.isfile(pjoin(path, 'x_trn.npy')):
			trn, vld, tst = _load(path, device)
		else:
			# get transforms
			kws_resize = dict(
				size=28,
				antialias=True,
				interpolation=F_vis.InterpolationMode.NEAREST,
			)
			transform = torchvision.transforms.Compose([
				torchvision.transforms.Resize(**kws_resize),
				InvertBinaryPILImage(),
				torchvision.transforms.ToTensor(),
				ToDevice(device=device),
			])
			# make dataset
			kws = dict(root=load_dir, transform=transform)
			trn = torchvision.datasets.Omniglot(background=True, **kws)
			vld = torchvision.datasets.Omniglot(background=False, **kws)
			# process and save
			trn, vld, _ = _process_and_save(
				trn=trn, vld=vld, save_dir=path)

	elif dataset == 'SVHN':
		defaults = dict(grey=True)
		kwargs = setup_kwargs(defaults, kwargs)
		path = 'processed' + ('_grey' if kwargs['grey'] else '')
		path = pjoin(load_dir, 'SVHN', path)
		if os.path.isfile(pjoin(path, 'x_trn.npy')):
			trn, vld, tst = _load(path, device)
		else:
			# get transforms
			transform = [
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Grayscale()
				if kwargs['grey'] else None,
				torchvision.transforms.Normalize(
					mean=(0.5,) * 3, std=(0.5,) * 3)
				if not kwargs['grey'] else None,
				ToDevice(device=device),
			]
			transform = torchvision.transforms.Compose([
				t for t in transform if t is not None
			])
			# make dataset
			kws = dict(
				root=pjoin(load_dir, 'SVHN'),
				transform=transform,
			)
			trn = torchvision.datasets.SVHN(split='train', **kws)
			vld = torchvision.datasets.SVHN(split='test', **kws)
			# process and save
			trn, vld, _ = _process_and_save(
				trn=trn, vld=vld, save_dir=path)

	elif dataset == 'CIFAR10':
		defaults = dict(grey=False, augment=True)
		kwargs = setup_kwargs(defaults, kwargs)
		path = 'processed' + ('_grey' if kwargs['grey'] else '')
		path = pjoin(load_dir, 'CIFAR10', path)
		if os.path.isfile(pjoin(path, 'x_trn.npy')):
			trn, vld, tst = _load(path, device)
		else:
			# get transforms
			transform_list = [
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Grayscale()
				if kwargs['grey'] else None,
				torchvision.transforms.Normalize(
					mean=(0.5,) if kwargs['grey'] else (0.5,) * 3,
					std=(0.5,) if kwargs['grey'] else (0.5,) * 3),
				ToDevice(device=device),
			]
			transform_list = [
				t for t in transform_list
				if t is not None
			]
			transform = torchvision.transforms.Compose(
				transform_list)
			# make dataset
			kws = dict(
				root=pjoin(load_dir, 'CIFAR10'),
				transform=transform,
			)
			trn = torchvision.datasets.CIFAR10(train=True, **kws)
			vld = torchvision.datasets.CIFAR10(train=False, **kws)
			if kwargs['augment']:
				flip = torchvision.transforms.RandomHorizontalFlip(p=1.0)
				transform = torchvision.transforms.Compose(
					[flip] + transform_list)
				kws['transform'] = transform
				trn_aug = torchvision.datasets.CIFAR10(train=True, **kws)
			else:
				trn_aug = None
			# process and save
			trn, vld, _ = _process_and_save(
				trn=trn,
				vld=vld,
				trn_aug=trn_aug,
				save_dir=path,
			)

	elif dataset == 'ImageNet32':
		defaults = dict(skip_trn=False)
		kwargs = setup_kwargs(defaults, kwargs)
		path = pjoin(load_dir, dataset, 'processed')
		if os.path.isfile(pjoin(path, 'x_trn.npy')):
			trn, vld, tst = _load(path, device, kwargs['skip_trn'])
		else:
			raise RuntimeError(dataset)

	elif dataset == 'CelebA':
		defaults = dict(grey=False)
		kwargs = setup_kwargs(defaults, kwargs)
		path = 'processed' + ('_grey' if kwargs['grey'] else '')
		path = pjoin(load_dir, dataset, path)
		if os.path.isfile(pjoin(path, 'x_trn.npy')):
			trn, vld, tst = _load(path, device)
		else:
			# get transforms
			transform_list = [
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Grayscale()
				if kwargs['grey'] else None,
				torchvision.transforms.Normalize(
					mean=(0.5,) if kwargs['grey'] else (0.5,) * 3,
					std=(0.5,) if kwargs['grey'] else (0.5,) * 3),
				ToDevice(device=device),
			]
			transform_list = [
				t for t in transform_list
				if t is not None
			]
			transform = torchvision.transforms.Compose(
				transform_list)
			# make dataset
			kws = dict(
				root=pjoin(load_dir, dataset),
				transform=transform,
			)
			trn = torchvision.datasets.CelebA(split='train', **kws)
			vld = torchvision.datasets.CelebA(split='valid', **kws)
			tst = torchvision.datasets.CelebA(split='test', **kws)
			# process and save
			trn, vld, tst = _process_and_save(
				trn=trn,
				vld=vld,
				tst=tst,
				save_dir=path,
			)

	elif dataset.startswith('BALLS'):
		defaults = dict(vld_split=0.25)
		kwargs = setup_kwargs(defaults, kwargs)

		# create load path
		path = pjoin(
			load_dir,
			'BALLS',
			f"npix-{int_from_str(dataset)}",
		)
		# load
		x_tst = np.load(pjoin(path, 'x_tst.npy'))
		z_tst = np.load(pjoin(path, 'z_tst.npy'))
		x = np.load(pjoin(path, 'x.npy'))
		z = np.load(pjoin(path, 'z.npy'))
		# split into trn / vld
		frac = 1 - kwargs['vld_split']
		idx = int(len(x) * frac)
		x_trn, x_vld = x[:idx], x[idx:]
		z_trn, z_vld = z[:idx], z[idx:]

		# to tensor
		x_trn, x_vld, x_tst, z_trn, z_vld, z_tst = map(
			_to_device_fun(device),
			[x_trn, x_vld, x_tst, z_trn, z_vld, z_tst],
		)
		# to dataset
		trn = torch.utils.data.TensorDataset(x_trn, z_trn)
		vld = torch.utils.data.TensorDataset(x_vld, z_vld)
		tst = torch.utils.data.TensorDataset(x_tst, z_tst)

	else:
		raise ValueError(dataset)

	return trn, vld, tst


def split_data(
		n_samples: int,
		n_blocks: int = 1,
		vld_portion: float = 0.2, ):
	assert 0 < vld_portion < 1
	indices = range(n_samples)
	block_size = len(indices) // n_blocks

	trn_inds, vld_inds = [], []
	for b in range(n_blocks):
		start = b * block_size
		if b == n_blocks - 1:
			end = len(indices)
		else:
			end = start + block_size

		block_inds = indices[start:end]
		vld_size = np.round(len(block_inds) * vld_portion)
		trn_size = len(block_inds) - int(vld_size)

		trn_inds.extend(block_inds[:trn_size])
		vld_inds.extend(block_inds[trn_size:])

	assert not set(trn_inds).intersection(
		vld_inds), "must be non-overlapping"
	trn_inds, vld_inds = map(
		lambda x: np.array(x),
		[trn_inds, vld_inds],
	)
	return trn_inds, vld_inds


def dataset_dims(dataset: str):
	channel_sz = 1
	dim_is_in_name = (
		any(d in dataset for d in [
			'vH16', 'vH32', 'CIFAR16',
			'Kyoto16', 'Kyoto32']) or
		dataset.startswith('BALLS')
	)
	if dim_is_in_name:
		pixel_sz = int_from_str(dataset)
	elif dataset.endswith('MNIST'):
		pixel_sz = 28
	elif dataset == 'Omniglot':
		pixel_sz = 28
	elif dataset == 'SVHN':
		pixel_sz = 32
	elif dataset in ['CIFAR10', 'ImageNet32']:
		pixel_sz = 32
		channel_sz = 3
	elif dataset == 'CelebA':
		pixel_sz = (218, 178)
		channel_sz = 3
	else:
		raise ValueError(dataset)
	if '-col' in dataset:
		channel_sz = 3
	if isinstance(pixel_sz, int):  # square
		pixel_sz = (pixel_sz, pixel_sz)
	dims = (channel_sz, *pixel_sz)
	return dims


class ToDevice:
	def __init__(self, device):
		self.device = device

	def __call__(self, x):
		return x.to(self.device)


class InvertBinaryPILImage(object):
	def __call__(self, img):
		# mode 'L': grayscale
		img = img.convert('L')
		return ImageOps.invert(img)


def _load(load_dir, device, skip_trn=False):
	# load numpy
	x_vld = np.load(pjoin(load_dir, 'x_vld.npy'))
	y_vld = np.load(pjoin(load_dir, 'y_vld.npy'))
	if skip_trn:
		x_trn = None
		y_trn = None
	else:
		x_trn = np.load(pjoin(load_dir, 'x_trn.npy'))
		y_trn = np.load(pjoin(load_dir, 'y_trn.npy'))
	if os.path.isfile(pjoin(load_dir, 'x_tst.npy')):
		x_tst = np.load(pjoin(load_dir, 'x_tst.npy'))
		y_tst = np.load(pjoin(load_dir, 'y_tst.npy'))
	else:
		x_tst, y_tst = None, None

	# to tensor
	x_trn, x_vld, y_trn, y_vld = map(
		_to_device_fun(device),
		[x_trn, x_vld, y_trn, y_vld],
	)
	if x_tst is not None:
		x_tst, y_tst = map(
			_to_device_fun(device),
			[x_tst, y_tst],
		)

	# dataset objects
	vld = torch.utils.data.TensorDataset(x_vld, y_vld)
	if skip_trn:
		trn = None
	else:
		trn = torch.utils.data.TensorDataset(x_trn, y_trn)
	if x_tst is not None:
		tst = torch.utils.data.TensorDataset(x_tst, y_tst)
	else:
		tst = None
	return trn, vld, tst


def _process_and_save(
		trn,
		vld,
		save_dir,
		tst=None,
		trn_aug=None,
		transpose=False, ):
	# process
	x_vld, y_vld = _process(vld)
	x_trn, y_trn = _process(trn)
	if tst is not None:
		x_tst, y_tst = _process(tst)
	else:
		x_tst, y_tst = None, None
	if trn_aug is not None:
		x_trn_aug, y_trn_aug = _process(trn_aug)
	else:
		x_trn_aug, y_trn_aug = None, None
	# transpose?
	if transpose:
		x_vld = torch.swapaxes(x_vld, 2, 3)
		x_trn = torch.swapaxes(x_trn, 2, 3)
		if tst is not None:
			x_tst = torch.swapaxes(x_tst, 2, 3)
		if trn_aug is not None:
			x_trn_aug = torch.swapaxes(x_trn_aug, 2, 3)
	# concat augmented data
	if trn_aug:
		x_trn = torch.cat([x_trn, x_trn_aug])
		y_trn = torch.cat([y_trn, y_trn_aug])
	# save
	os.makedirs(save_dir, exist_ok=True)
	_save = {
		'x_trn': tonp(x_trn),
		'y_trn': tonp(y_trn),
		'x_vld': tonp(x_vld),
		'y_vld': tonp(y_vld),
	}
	if x_tst is not None:
		_save = {
			**_save,
			'x_tst': tonp(x_tst),
			'y_tst': tonp(y_tst),
		}
	for name, obj in _save.items():
		save_obj(
			obj=obj,
			file_name=name,
			save_dir=save_dir,
			verbose=True,
			mode='npy',
		)
	# to dataset
	trn = torch.utils.data.TensorDataset(x_trn, y_trn)
	vld = torch.utils.data.TensorDataset(x_vld, y_vld)
	if x_tst is not None:
		tst = torch.utils.data.TensorDataset(x_tst, y_tst)
	return trn, vld, tst


def _process(dataset):
	loader = torch.utils.data.DataLoader(
		dataset, batch_size=len(dataset))
	return next(iter(loader))


def _to_device_fun(
		device: torch.device,
		dtype=torch.float, ):
	def _fun(a):
		if a is None:
			return
		return torch.tensor(
			data=a,
			dtype=dtype,
			device=device,
		)
	return _fun
