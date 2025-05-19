from utils.generic import *
from base.helper import ParameterGrid
from main.datafeeder import FEEDER_CLASSES


def drifting_grating_feeder(
		npix: int,
		param_grid: ParameterGrid,
		device: str | torch.device,
		n_frames_per_cycle: int = 500, ):
	param_vectors = param_grid.get_param_vectors()
	feeder_kws = dict(
		npix=npix,
		#
		theta=param_vectors['theta'],
		s_freq=param_vectors['s_freq'],
		t_freq=1 / n_frames_per_cycle,
		contrast=param_vectors['contrast'],
		normalize_coords=True,
		#
		device=device,
		flatten=True,
	)
	feeder = FEEDER_CLASSES['drifting-grating'](**feeder_kws)
	return feeder, feeder_kws


def drifting_grating_feeder_params(
		theta: Sequence[float] = None,
		contrast: Sequence[float] = None,
		s_freq: Sequence[float] = None,
		theta_min: float = 0.0,
		theta_max: float = 180.0,
		theta_delta: float = 5.0,
		contrast_min: float = 0.5,
		contrast_max: float = 1.0,
		contrast_delta: float = 0.5,
		s_freq_min: float = 0.5,
		s_freq_max: float = 8.0,
		s_freq_delta: float = 0.5, ) -> ParameterGrid:
	# orientation
	if theta is None:
		theta_num = (theta_max - theta_min) / theta_delta
		theta_num = int(np.ceil(theta_num)) + 1
		theta = np.linspace(
			start=theta_min,
			stop=theta_max,
			num=theta_num,
		)
	# contrast
	if contrast is None:
		contrast_num = (contrast_max - contrast_min) / contrast_delta
		contrast_num = int(np.ceil(contrast_num)) + 1
		contrast = np.linspace(
			start=contrast_min,
			stop=contrast_max,
			num=contrast_num,
		)
	# spatial frequency
	if s_freq is None:
		s_freq_num = (s_freq_max - s_freq_min) / s_freq_delta
		s_freq_num = int(np.ceil(s_freq_num)) + 1
		s_freq = np.linspace(
			start=s_freq_min,
			stop=s_freq_max,
			num=s_freq_num,
		)
	# put together
	params = {
		'contrast': np.round(contrast, decimals=2),
		's_freq': np.round(s_freq, decimals=2),
		'theta': np.round(theta, decimals=2),
	}
	return ParameterGrid(params)
