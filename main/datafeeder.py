from base.helper import rotate_img, get_rng, torch, F, np
from typing import *
import math


class DataFeeder:
	def __init__(
			self,
			x: torch.Tensor = None,
			flatten: bool = True,
			seed: int = None,
			**kwargs,
	):
		if x is not None:
			if isinstance(x, np.ndarray):
				x = torch.from_numpy(x)
			while x.ndim < 4:
				x = x.unsqueeze(0)
		self.x = x  # shape: [B, C, H, W]
		self.rng = get_rng(seed)
		self.flatten = flatten

	def __len__(self):
		return len(self.x)

	def __getitem__(self, t):
		return self.get_frame(t)

	def get_frame(self, t):
		raise NotImplementedError


class StationaryFeeder(DataFeeder):

	def get_frame(self, t):
		if self.flatten:
			return self.x.flatten(start_dim=1)
		return self.x


class SequenceFeeder(DataFeeder):

	def get_frame(self, t):
		# assumes shape is [B, T, ...]
		try:
			x = self.x[:, t]
			if self.flatten:
				x = x.flatten(start_dim=1)
			return x
		except IndexError:
			msg = f"invalid t={t}, shape: {self.x.shape}"
			raise ValueError(msg)


class RotateFeeder(DataFeeder):
	def __init__(
			self,
			x: torch.Tensor,
			theta_0: float = None,
			delta_theta: float = 8.0,
			**kwargs, ):
		super(RotateFeeder, self).__init__(
			x=x, **kwargs)
		if theta_0 is None:
			theta_0 = self.rng.uniform(0, 360)
		self.theta_0 = float(theta_0)
		self.delta_theta = float(delta_theta)

	def get_frame(self, t):
		current_angle = (
			self.theta_0 +
			self.delta_theta * t
		)
		x = rotate_img(
			x=self.x,
			angles=current_angle,
			interpolation='bilinear',
			hist_match=True,
		)
		if self.flatten:
			x = x.flatten(start_dim=1)
		return x


class JitterFeeder(DataFeeder):
	def __init__(
		self,
		x: torch.Tensor,
		mode: str = 'pad',
		diff_coeff: float = 0.6,
		harmonic_strength: float = 0.03,
		initial_pos: tuple = (0.0, 0.0),
		crop_size: tuple = None,
		**kwargs,
	):
		"""
		Args:
			x: Batch of images with shape [B, C, H, W].
			mode: Either 'pad' or 'crop'.
			diff_coeff: Diffusion coefficient. For each
				axis, the displacement per time step is
				sampled as N(0, sqrt(diff_coeff)).
			crop_size: Tuple (crop_h, crop_w).
				Needed when mode == 'crop'.
			initial_pos: Initial displacement as a tuple
				in pixel units.
		"""
		super(JitterFeeder, self).__init__(
			x=x, **kwargs)
		self.mode = mode
		self.diff_coeff = diff_coeff
		self.harmonic_strength = harmonic_strength
		# Current pos: tensor in pixel coordinates
		self.current_pos = torch.tensor(
			initial_pos, dtype=torch.float)
		self.trajectory = [self.current_pos.clone()]

		if self.mode not in ['pad', 'crop']:
			raise ValueError("Choices: {'pad', 'crop'}")

		if self.mode == 'crop':
			assert crop_size is not None, \
				"must provide crop_size when mode is 'crop'"
			if isinstance(crop_size, (float, int)):
				crop_size = (float(crop_size), ) * 2
			self.crop_size = crop_size
			crop_h, crop_w = crop_size
			# max allowed displacement
			_, _, h, w = self.x.shape
			self.max_disp_y = (h - crop_h) // 2
			self.max_disp_x = (w - crop_w) // 2

	def get_frame(self, t):
		"""
		Update the current position by adding a
		Brownian increment and then apply the
		corresponding jitter transformation.
		"""
		delta_np = self.rng.normal(
			0, math.sqrt(self.diff_coeff), size=(2,))
		delta = torch.tensor(delta_np, dtype=torch.float)

		force = -self.harmonic_strength * self.current_pos
		self.current_pos = self.current_pos + delta + force
		self.trajectory.append(self.current_pos.clone())

		dx, dy = self.current_pos.tolist()
		b, _, h, w = self.x.shape

		if self.mode == 'crop':
			dx = max(
				-self.max_disp_x,
				min(self.max_disp_x, dx),
			)
			dy = max(
				-self.max_disp_y,
				min(self.max_disp_y, dy),
			)
			self.current_pos = torch.tensor(
				[dx, dy], dtype=torch.float)

			crop_h, crop_w = self.crop_size
			# Define the center of the image.
			center_y = h // 2
			center_x = w // 2
			# Compute the top-left coordinates of the crop.
			start_y = center_y - crop_h // 2 + int(round(dy))
			start_x = center_x - crop_w // 2 + int(round(dx))
			end_y = start_y + crop_h
			end_x = start_x + crop_w
			start_x, start_y, end_x, end_y = map(
				int, [start_x, start_y, end_x, end_y])
			# Extract the crop from each image in the batch.
			frame = self.x[:, :, start_y:end_y, start_x:end_x]

		elif self.mode == 'pad':
			# Convert pixel displacement to normalized coordinates.
			# (When align_corners=True, a translation of dx pixels
			# corresponds to tx = dx*2/W)
			tx = dx * 2 / w
			ty = dy * 2 / h

			# Create the affine transformation matrix for each image.
			# The matrix has shape (2, 3): [ [1, 0, tx], [0, 1, ty] ]
			theta = [
				[1, 0, tx],
				[0, 1, ty],
			]
			theta = torch.tensor(
				theta,
				dtype=torch.float,
				device=self.x.device,
			)  # shape (2, 3)
			# Expand theta to the full batch size.
			theta = theta.unsqueeze(0).repeat(b, 1, 1)  # (b, 2, 3)

			# Generate the sampling grid.
			grid = F.affine_grid(
				theta=theta,
				size=list(self.x.size()),
				align_corners=True,
			)
			# Apply the grid sampling (w/ zero padding)
			frame = F.grid_sample(
				input=self.x,
				grid=grid,
				padding_mode='zeros',
				align_corners=True,
			)
		else:
			raise ValueError(f"Invalid mode: {self.mode}")

		if self.flatten:
			frame = frame.flatten(start_dim=1)
		return frame

	def reset_trajectory(self):
		self.trajectory = [self.current_pos.clone()]
		return


class DriftingGratingFeeder(DataFeeder):
	def __init__(
			self,
			npix: int | Tuple[int, int],
			theta: float | Iterable[float],
			s_freq: float | Iterable[float],
			t_freq: float | Iterable[float],
			contrast: float | Iterable[float],
			device: torch.device = 'cuda',
			normalize_coords: bool = True,
			**kwargs,
	):
		"""
		:param npix:
		:param theta:
		:param s_freq:
		:param t_freq: e.g., to see a full cycle
			per 10 frames, use t_freq = 0.1
		:param contrast:
		:param device:
		:param normalize_coords:
			With normalized coordinates (grid spanning [-0.5, 0.5]),
			a spatial frequency of 1 produces 1 cycle across the image.
			e.g., to see 2 cycles per image, you should set s_freq = 2
		"""
		super(DriftingGratingFeeder, self).__init__(**kwargs)

		if isinstance(npix, int):
			npix = (npix, npix)
		h, w = npix

		theta, s_freq, t_freq, contrast = map(
			_get_fun1(device), [theta, s_freq, t_freq, contrast])
		self.batch_size = max(
			len(theta), len(s_freq),
			len(t_freq), len(contrast),
		)
		self.theta, self.s_freq, self.t_freq, self.contrast = map(
			_get_fun2(self.batch_size), [theta, s_freq, t_freq, contrast]
		)

		theta_rad = torch.deg2rad(self.theta)
		self.cos_theta = torch.cos(theta_rad)
		self.sin_theta = torch.sin(theta_rad)

		if normalize_coords:
			xs = torch.linspace(-0.5, 0.5, steps=w)
			ys = torch.linspace(-0.5, 0.5, steps=h)
		else:
			xs = torch.linspace(-w / 2, w / 2, steps=w)
			ys = torch.linspace(-h / 2, h / 2, steps=h)
		ys, xs = torch.meshgrid(ys, xs, indexing='ij')
		self.xs = xs.to(device=device)  # shape: (h, w)
		self.ys = ys.to(device=device)  # shape: (h, w)

	def __len__(self):
		return self.batch_size

	def get_frame(self, t):
		# Expand spatial grid: (H, W) → (1, H, W)
		xs = self.xs.unsqueeze(0)
		ys = self.ys.unsqueeze(0)
		# Expand cos_theta and sin_theta: (B, ) → (B, 1, 1)
		cos_theta = self.cos_theta.unsqueeze(1).unsqueeze(2)
		sin_theta = self.sin_theta.unsqueeze(1).unsqueeze(2)
		# Spatial index
		s = xs * cos_theta + ys * sin_theta

		# Expand spatial and temporal freqs → (B, 1, 1)
		arg = 2 * math.pi * (
			self.s_freq.unsqueeze(1).unsqueeze(2) * s -
			self.t_freq.unsqueeze(1).unsqueeze(2) * t
		)

		# Expand contrast → (B, 1, 1)
		contrast = self.contrast.unsqueeze(1).unsqueeze(2)
		pattern = contrast * torch.sin(arg)  # shape: (B, H, W)

		# Add a channel dimension → (B, 1, H, W)
		frame = pattern.unsqueeze(1)

		if self.flatten:
			frame = frame.flatten(start_dim=1)
		return frame


class WhiteNoiseFeeder(DataFeeder):
	def __init__(
			self,
			size: int | List[int],
			n_repeats: int = 1,
			batch_size: int = 1000,
			noise_mean: float = 0.0,
			noise_sigma: float = 1.0,
			device: torch.device = 'cuda',
			**kwargs,
	):
		super(WhiteNoiseFeeder, self).__init__(**kwargs)

		self.n_repeats = n_repeats
		self.batch_size = batch_size
		self.device = device

		if isinstance(size, int):
			size = (1, size, size)
		elif len(size) == 2:
			size = (1, *size)
		else:
			assert len(size) == 3
		self.noise_kws = dict(
			loc=noise_mean,
			scale=noise_sigma,
			size=(batch_size, *size),
		)
		self.current_frame = None
		self._sample_frame()

	def __len__(self):
		return self.batch_size

	def _sample_frame(self):
		self.current_frame = torch.tensor(
			data=self.rng.normal(**self.noise_kws),
			device=self.device,
			dtype=torch.float,
		)
		return

	def get_frame(self, t):
		if t % self.n_repeats == 0:
			self._sample_frame()
		frame = self.current_frame
		if self.flatten:
			frame = frame.flatten(start_dim=1)
		return frame


def _get_fun1(device, dtype=torch.float):
	def _fun1(item):
		if not isinstance(item, Iterable):
			item = [item]
		item = torch.tensor(
			data=item,
			dtype=dtype,
			device=device,
		)
		return item
	return _fun1


def _get_fun2(batch_size):
	def _fun2(item):
		assert isinstance(item, torch.Tensor)
		item = item.expand((batch_size, ))
		return item
	return _fun2


FEEDER_CLASSES = {
	'stationary': StationaryFeeder,
	'sequence': SequenceFeeder,
	'jitter': JitterFeeder,
	'rotate': RotateFeeder,
	'white-noise': WhiteNoiseFeeder,
	'drifting-grating': DriftingGratingFeeder,
}
