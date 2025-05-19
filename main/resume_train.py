from utils.generic import *
from .train import load_model_main, save_fit_info


def _setup_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"device",
		choices=range(torch.cuda.device_count()),
		help='cuda:device',
		type=int,
	)
	parser.add_argument(
		"model_name",
		help='model string?',
		type=str,
	)
	parser.add_argument(
		"fit_name",
		help='fit string?',
		type=str,
	)
	parser.add_argument(
		"--dry_run",
		help='to make sure config is alright',
		action='store_true',
		default=False,
	)
	parser.add_argument(
		"--dont_save_info",
		help='saves info by default',
		action='store_true',
		default=False,
	)
	parser.add_argument(
		"--cudnn_bench",
		help='use cudnn benchmark?',
		action='store_true',
		default=False,
	)
	parser.add_argument(
		"--verbose",
		help='to make sure config is alright',
		action='store_true',
		default=False,
	)
	return parser.parse_args()


def _main():
	args = _setup_args()
	device = f"cuda:{args.device}"

	tr, meta = load_model_main(
		model_name=args.model_name,
		fit_name=args.fit_name,
		checkpoint=-1,
		device=device,
		shuffle=True,
		strict=True,
	)
	epochs = range(  # remaining epochs
		meta['checkpoint'],
		tr.cfg.epochs,
	)
	if not len(epochs):
		return

	args.archi = tr.model.cfg.attr2archi()

	if args.cudnn_bench:
		torch.backends.cudnn.benchmark = True
		torch.backends.cudnn.benchmark_limit = 0

	if args.verbose:
		print(args)
		tr.model.print()
		print(f"# train iters: {tr.n_iters:,}\n")
		tr.print()
		print()

	start = now(True)
	if not args.dry_run:
		tr.train(epochs=epochs, fresh_fit=False)
		if not args.dont_save_info:
			save_fit_info(
				tr=tr,
				args=vars(args),
				start=start,
			)
	return


if __name__ == "__main__":
	_main()
