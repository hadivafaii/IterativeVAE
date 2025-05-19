from utils.generic import *


def copy_fitted_model(
		dst_dir: str,
		chckpt_dir: str,
		dst_name: str = None,
		checkpoint: int = -1,
		overwrite: bool = False, ):

	chckpt_dir = add_home(chckpt_dir)
	parent_dir = os.path.dirname(chckpt_dir)
	dst_name = dst_name if dst_name else \
		os.path.basename(parent_dir)
	dst_dir = pjoin(dst_dir, dst_name)
	os.makedirs(dst_dir, exist_ok=True)

	to_copy = {}

	# config: model
	files = sorted(os.listdir(parent_dir))
	config_model = next(
		f for f in files if
		f.endswith('.json')
	)
	to_copy[config_model] = pjoin(
		parent_dir, config_model)
	# config: trainer
	files = sorted(os.listdir(chckpt_dir))
	config_trainer = next(
		f for f in files if
		f.endswith('.json')
	)
	to_copy[config_trainer] = pjoin(
		chckpt_dir, config_trainer)
	# checkpoint file
	fname_pt = [
		f for f in files if
		f.split('.')[-1] == 'pt'
	]
	if checkpoint == -1:
		fname_pt = fname_pt[-1]
	else:
		fname_pt = next(
			f for f in fname_pt if
			checkpoint == _chkpt(f)
		)
	to_copy[fname_pt] = pjoin(
		chckpt_dir, fname_pt)

	for name, src in to_copy.items():
		dst = pjoin(dst_dir, name)
		exists = os.path.isfile(dst)
		if overwrite or not exists:
			shutil.copy(src, dst)
	return dst_dir


def copy_checkpoints(
		project: str,
		pattern: str = "[INFO] Checkpoint Directory:",
		recursive: bool = False, ):
	dst = pjoin('Dropbox/chkpts', project)
	src = pjoin('Dropbox/git', project, 'logs')
	dst, src = map(add_home, [dst, src])
	if recursive:
		txt_files = pathlib.Path(src).rglob('*.txt')
	else:
		txt_files = pathlib.Path(src).glob('*.txt')
	tot = 0
	for f in txt_files:
		chckpt_dir = _find_line(str(f), pattern)
		if not os.path.isdir(chckpt_dir):
			continue
		# copy model checkpoint
		dst_name = f.name.replace('.txt', '')
		dst_dir = copy_fitted_model(
			dst_dir=dst,
			chckpt_dir=chckpt_dir,
			dst_name=dst_name,
		)
		# copy txt log for the record
		shutil.copy(str(f), pjoin(dst_dir, f.name))
		tot += 1
	return tot


def _chkpt(f):
	return int(f.split('_')[0].split('-')[-1])


def _find_line(f: str, pattern: str):
	result = None
	with open(f, 'r') as file:
		content = file.readlines()
	for line in content:
		if pattern in line:
			idx = content.index(line) + 1
			result = content[idx].strip()
			break
	return result


def _setup_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"project",
		help='current project name',
		type=str,
	)
	return parser.parse_args()


def _main():
	args = _setup_args()
	tot = copy_checkpoints(args.project)
	msg = ' ——— '.join([
		f"[PROGRESS] a total of {tot} models copied",
		f"host: {os.uname().nodename}"
	])
	print(msg)
	return


if __name__ == "__main__":
	_main()
