import argparse
import os, json
from multiprocessing import cpu_count

from datasets import preprocessor
from hparams import hparams
from tqdm import tqdm


def preprocess_vctk(args, input_folders, output_folder, hparams):
	out_dir = os.path.join(output_folder, args.dataset)
	mel_dir = os.path.join(out_dir, 'mels')
	wav_dir = os.path.join(out_dir, 'audio')
	linear_dir = os.path.join(out_dir, 'linear')
	os.makedirs(mel_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(linear_dir, exist_ok=True)

	metadata = preprocessor.build_from_path_vctk(hparams, input_folders, mel_dir, linear_dir, wav_dir,
												 args.n_jobs, tqdm=tqdm)
	write_metadata(args.dataset, metadata, out_dir)


def preprocess_THCHS(args, input_folders, output_folder, hparams):
	out_dir = os.path.join(output_folder, args.dataset)
	mel_dir = os.path.join(out_dir, 'mels')
	wav_dir = os.path.join(out_dir, 'audio')
	linear_dir = os.path.join(out_dir, 'linear')
	os.makedirs(mel_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(linear_dir, exist_ok=True)

	metadata = preprocessor.build_from_path_THCHS(hparams, input_folders, mel_dir, linear_dir, wav_dir, args.n_jobs, tqdm=tqdm)
	write_metadata(args.dataset, metadata, out_dir)


def preprocess(args, input_folders, out_dir, hparams):
	out_dir = os.path.join(out_dir, args.dataset)
	mel_dir = os.path.join(out_dir, 'mels')
	wav_dir = os.path.join(out_dir, 'audio')
	linear_dir = os.path.join(out_dir, 'linear')
	os.makedirs(out_dir, exist_ok=True)
	os.makedirs(mel_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(linear_dir, exist_ok=True)

	metadata = preprocessor.build_from_path_ljspeech_and_Mailab(hparams, input_folders, mel_dir, linear_dir, wav_dir, args.n_jobs, tqdm=tqdm)
	write_metadata(args.dataset, metadata, out_dir)


def write_metadata(dataset, metadata, out_dir):
	with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
		for m in metadata:
			f.write('|'.join([str(x) for x in m]) + '\n')
	mel_frames = sum([int(m[4]) for m in metadata])
	timesteps = sum([int(m[3]) for m in metadata])
	sr = hparams.sample_rate
	hours = timesteps / sr / 3600
	print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(
		len(metadata), mel_frames, timesteps, hours))
	print('Max input length (text chars): {}'.format(max(len(m[5]) for m in metadata)))
	print('Max mel frames length: {}'.format(max(int(m[4]) for m in metadata)))
	print('Max audio timesteps length: {}'.format(max(m[3] for m in metadata)))

	try:
		with open('train_data_dict.json', 'r') as f:
			train_data_dict = json.load(f)
	except:
		train_data_dict = {}
	train_data_dict[dataset] = os.path.join(out_dir, 'train.txt')
	with open('train_data_dict.json', 'w') as f:
		json.dump(train_data_dict, f)



def norm_data(args):

	merge_books = (args.merge_books=='True')

	print('Selecting data folders..')
	#supported_datasets = ['LJSpeech-1.0', 'LJSpeech-1.1', 'M-AILABS']
	#if args.dataset not in supported_datasets:
	#	raise ValueError('dataset value entered {} does not belong to supported datasets: {}'.format(
	#		args.dataset, supported_datasets))

	if args.dataset == 'ljspeech':
		return [args.data_path]


	if args.dataset == 'M-AILABS':
		supported_languages = ['en_US', 'en_UK', 'fr_FR', 'it_IT', 'de_DE', 'es_ES', 'ru_RU',
			'uk_UK', 'pl_PL', 'nl_NL', 'pt_PT', 'fi_FI', 'se_SE', 'tr_TR', 'ar_SA']
		if args.language not in supported_languages:
			raise ValueError('Please enter a supported language to use from M-AILABS dataset! \n{}'.format(
				supported_languages))

		supported_voices = ['female', 'male', 'mix']
		if args.voice not in supported_voices:
			raise ValueError('Please enter a supported voice option to use from M-AILABS dataset! \n{}'.format(
				supported_voices))

		path = os.path.join(args.data_path, args.language, 'by_book', args.voice)
		supported_readers = [e for e in os.listdir(path) if os.path.isdir(os.path.join(path,e))]
		if args.reader not in supported_readers:
			raise ValueError('Please enter a valid reader for your language and voice settings! \n{}'.format(
				supported_readers))

		path = os.path.join(path, args.reader)
		supported_books = [e for e in os.listdir(path) if os.path.isdir(os.path.join(path,e))]
		if merge_books:
			return [os.path.join(path, book) for book in supported_books]

		else:
			if args.book not in supported_books:
				raise ValueError('Please enter a valid book for your reader settings! \n{}'.format(
					supported_books))

			return [os.path.join(path, args.book)]

	if args.dataset == 'vctk':
		return args.data_path
	if args.dataset == 'THCHS':
		return args.data_path





def run_preprocess(args, hparams):
	input_folders = norm_data(args)
	output_folder = os.path.join(args.base_dir, args.output)

	if args.dataset == 'ljspeech':
		preprocess(args, input_folders, output_folder, hparams)
	if args.dataset == 'M-AILABS':
		args.dataset = args.dataset + '_' + args.reader
		preprocess(args, input_folders, output_folder, hparams)
	if args.dataset == 'vctk':
		preprocess_vctk(args, input_folders, output_folder, hparams)
	if args.dataset == 'THCHS':
		preprocess_THCHS(args, input_folders, output_folder, hparams)



def main():
	print('initializing preprocessing..')
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', default='')
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--language', default='en_US')
	parser.add_argument('--voice', default='female')
	parser.add_argument('--reader', default='mary_ann')
	parser.add_argument('--merge_books', default='True')
	parser.add_argument('--book', default='northandsouth')
	parser.add_argument('--output', default='training_data')
	parser.add_argument('--n_jobs', type=int, default=cpu_count())
	parser.add_argument('--dataset', type=str, choices=['THCHS', 'aishell', 'ljspeech', 'M-AILABS', 'vctk', 'THCHS'])
	parser.add_argument('--data_path', type=str, default='data/LJSpeech-1.1')

	args = parser.parse_args()

	modified_hp = hparams.parse(args.hparams)

	assert args.merge_books in ('False', 'True')

	run_preprocess(args, modified_hp)


if __name__ == '__main__':
	main()
'''
python3 preprocess.py --dataset ljspeech --data_path 'data/LJSpeech-1.1'
python3 preprocess.py --dataset M-AILABS --language en_US --voice female --reader mary_ann --merge_books True --data_path data
python3 preprocess.py --dataset M-AILABS --language en_US --voice female --reader judy_bieber --merge_books True --data_path data
python3 preprocess.py --dataset M-AILABS --language en_US --voice male --reader elliot_miller --merge_books True --data_path data
python3 preprocess.py --dataset M-AILABS --language en_UK --voice female --reader elizabeth_klett --merge_books True --data_path data
python3 preprocess.py --dataset vctk --data_path data/VCTK-Corpus
python3 preprocess.py --dataset THCHS --data_path data/THCHS

'''
