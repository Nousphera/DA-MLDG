import os
import argparse
import pickle
import tensorflow as tf
import utils
from architectures import resnet18, rsc, da_mldg, mldg
tf.get_logger().setLevel('ERROR')

parser = argparse.ArgumentParser(description="Meta-Learning SER")
parser.add_argument("--runs", type=int, default=1, help="total runs")
parser.add_argument("--epochs", type=int, default=100, help="number of total epochs to run")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--learning_rate", type=float,  default=1e-3, help="learning rate")
parser.add_argument("--verbose", type=int, default=1, help="set verbosability level")
parser.add_argument("--data_dir", type=str, default=f"{os.environ['TFDS_DATA_DIR']}/raw/", help="store dir for datasets")
parser.add_argument("--num_bins", type=int, default=64, help="number of mel bins")
parser.add_argument("--input_length_in_second", type=float, default=3.5, help="length of audio input")
parser.add_argument("--scenario", type=str, default="pacs", help="scenario name to execute")
parser.add_argument("--method", type=str, default="deep_all", help="method name to execute")
parser.add_argument("--validation_freq", type=int, default=5, help="validation data check update")
parser.add_argument("--save_dir", type=str, default='./assets', help="save dir for model weights")
parser.add_argument("--result_dir", type=str, default='./assets', help="save dir for results")
parser.add_argument("--weights", type=str, default='./weights', help="path to pretrained weights")
args = parser.parse_args()

_save_name = lambda x: f"{args.scenario}_{args.method}_run_{x[0]}_{x[1]:0.4f}.h5"
_result_dir = lambda x: os.path.join(args.result_dir,f"rsc_{args.scenario}_{args.method}_run_{x}.pkl")
_use_logits = False if args.method=='rsc' else True

def get_model(input_shape, num_classes, num_domains):
	if args.method=='deepall':
		return resnet18.ResNet18.model(input_shape=input_shape[1:], num_classes=num_classes, weights=args.weights)
	elif args.method=='rsc':
		return rsc.RSC(input_shape=(None,)+input_shape[2:], num_classes=num_classes, weights=args.weights)
	elif args.method=='mldg':
		return mldg.MLDG(input_shape=input_shape[1:], num_classes=num_classes, num_domains=num_domains, weights=args.weights)
	elif args.method=='damldg':
		return da_mldg.DA_MLDG(input_shape=input_shape[1:], num_classes=num_classes, num_domains=num_domains, weights=args.weights)
	else:
		raise NotImplementedError(f'{args.method} is not Implemented')

def prepare_data(target_domain='art_painting',):

	drop_idx = True if args.method in ['deepall','rsc'] else False

	if args.scenario=='pacs':
		ds, num_classes, num_samples, num_domains = utils.Datasets.load_image_datasets(args.scenario, fp=args.data_dir)
		train_set = ds.keys()-[target_domain]
		print(f"Training on {train_set} and target domain is {target_domain} ({args.method})")
		train_ds = utils.ImageUtils.train_prep(ds, domains=train_set, bs=args.batch_size, drop_idx=drop_idx)
		test_ds = utils.ImageUtils.test_prep(ds, domains=[target_domain], bs=args.batch_size)
	else:
		ds, num_classes, num_samples, num_domains = utils.Datasets.load_audio_datasets(args.scenario, fp=args.data_dir)
		train_set = ds.keys()-[target_domain]
		print(f"Training on {train_set} and evaluating on {target_domain} ({args.method})")
		train_ds = utils.AudioUtils.train_prep(ds, domains=train_set, bins=args.num_bins, seconds=args.input_length_in_second, bs=args.batch_size, drop_idx=drop_idx)
		test_ds = utils.AudioUtils.test_prep(ds, domains=[target_domain], bins=args.num_bins)

	return (train_ds,test_ds),num_classes,num_domains

def main(id):
	tf.keras.backend.clear_session()
	results = {}
	if args.scenario in ['pacs','eng4']:
		for target_domain in utils.SCENARIOS[args.scenario]:
			(train_ds,test_ds), num_classes, num_domains = prepare_data(target_domain=target_domain)
			model = get_model(train_ds.element_spec[0].shape, num_classes, num_domains)
			model.compile(optimizer=utils.opt_fn((args.method, args.learning_rate)), loss=utils.loss_fn((_use_logits,'loss')), metrics=[utils.metric('acc')])
			model.fit(train_ds, epochs=args.epochs, validation_data=test_ds, validation_freq=args.validation_freq, verbose=args.verbose)
			metrics = model.evaluate(test_ds, verbose=0)
			print(f"[Evaluation] Loss: {metrics[0]:0.3f} - Accuracy: {100*metrics[1]:0.2f}\n ({args.method})")
			model.save_weights(os.path.join(args.save_dir,_save_name((id,metrics[1]))))
			results[target_domain]= metrics[1]
			tf.keras.backend.clear_session()
	else:
		target_domain = utils.SCENARIOS[args.scenario][-1]
		(train_ds,test_ds), num_classes, num_domains = prepare_data(target_domain=target_domain)
		model = get_model(train_ds.element_spec[0].shape, num_classes, num_domains)
		model.compile(optimizer=utils.opt_fn((args.method, args.learning_rate)), loss=utils.loss_fn((_use_logits,'loss')), metrics=[utils.metric('acc')])
		model.fit(train_ds, epochs=args.epochs, validation_data=test_ds, validation_freq=args.validation_freq, verbose=args.verbose)
		metrics = model.evaluate(test_ds, verbose=0)
		print(f"[Evaluation] Loss: {metrics[0]:0.3f} - Accuracy: {100*metrics[1]:0.2f} ({args.method})\n")
		model.save_weights(os.path.join(args.save_dir,_save_name((id,metrics[1]))))
		results[target_domain]= metrics[1]
		return results

if __name__ == "__main__":

	accuracies = {}
	parsed_args = ' '.join(f'{k}={v}\n' for k, v in vars(args).items())
	print("\nParameters:\n",parsed_args)
	#for run_id in range(args.runs):
	accuracies[f"{args.scenario}_{args.method}_run_{args.runs}"] = main(id=args.runs)
	with open(_result_dir(args.runs), 'wb') as handle:
		pickle.dump(accuracies, handle, protocol=pickle.HIGHEST_PROTOCOL)


