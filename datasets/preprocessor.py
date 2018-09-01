import os, re
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from datasets import audio
from wavenet_vocoder.util import is_mulaw, is_mulaw_quantize, mulaw, mulaw_quantize


def build_from_path_ljspeech_and_Mailab(hparams, input_dirs, mel_dir, linear_dir, wav_dir, n_jobs=12, tqdm=lambda x: x):
	"""
	Preprocesses the speech dataset from a gven input path to given output directories

	Args:
		- hparams: hyper parameters
		- input_dir: input directory that contains the files to prerocess
		- mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
		- linear_dir: output directory of the preprocessed speech linear-spectrogram dataset
		- wav_dir: output directory of the preprocessed speech audio dataset
		- n_jobs: Optional, number of worker process to parallelize across
		- tqdm: Optional, provides a nice progress bar

	Returns:
		- A list of tuple describing the train examples. this should be written to train.txt
	"""

	# We use ProcessPoolExecutor to parallelize across processes, this is just for
	# optimization purposes and it can be omited
	executor = ProcessPoolExecutor(max_workers=n_jobs)
	futures = []
	index = 1
	for input_dir in input_dirs:
		with open(os.path.join(input_dir, 'metadata.csv'), encoding='utf-8') as f:
			for line in f:
				parts = line.strip().split('|')
				basename = parts[0]
				wav_path = os.path.join(input_dir, 'wavs', '{}.wav'.format(basename))
				text = parts[2]
				futures.append(executor.submit(partial(_process_utterance, mel_dir, linear_dir, wav_dir, basename, wav_path, text, hparams, 1)))
				index += 1

	return [future.result() for future in tqdm(futures) if future.result() is not None]



def build_from_path_vctk(hparams, input_dir, mel_dir, linear_dir, wav_dir, n_jobs=12, tqdm=lambda x: x):
	"""
	Preprocesses the speech dataset from a gven input path to given output directories

	Args:
		- hparams: hyper parameters
		- input_dir: input directory that contains the files to prerocess
		- mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
		- linear_dir: output directory of the preprocessed speech linear-spectrogram dataset
		- wav_dir: output directory of the preprocessed speech audio dataset
		- n_jobs: Optional, number of worker process to parallelize across
		- tqdm: Optional, provides a nice progress bar

	Returns:
		- A list of tuple describing the train examples. this should be written to train.txt
	"""

	# We use ProcessPoolExecutor to parallelize across processes, this is just for
	# optimization purposes and it can be omited

	executor = ProcessPoolExecutor(max_workers=n_jobs)
	futures = []
	ct = 1
	input_dir = './' + input_dir
	vctk_path_wav = os.path.join(input_dir, 'wav48')
	vctk_path_txt = os.path.join(input_dir, 'txt')

	for dir_name in os.listdir(vctk_path_wav):
		# dir_name = 'p300'
		# wav_dir  = '/home/pattern/songjinming/tts/data/vctk-Corpus/wav48/p300'
		# txt_dir  = '/home/pattern/songjinming/tts/data/vctk-Corpus/txt/p300'
		wav_dir2 = os.path.join(vctk_path_wav, dir_name)
		txt_dir = os.path.join(vctk_path_txt, dir_name)
		for wav_file in os.listdir(wav_dir2):
			# wav_file  = 'p300_224.wav'
			# name_file = 'p300_224'
			# txt_file  = 'p300_224.txt'
			# wav_root  = '/home/pattern/songjinming/tts/data/vctk-Corpus/wav48/p300/p300_224.wav'
			# txt_root  = '/home/pattern/songjinming/tts/data/vctk-Corpus/txt/p300/p300_224.txt'
			name_file = os.path.splitext(wav_file)[0]
			basename = name_file
			# some file is not wav file and just skip
			# print(os.path.splitext(wav_file))
			if not os.path.splitext(wav_file)[1] == '.wav':
				continue
			txt_file = '.'.join([name_file, 'txt'])
			wav_path = os.path.join(wav_dir2, wav_file)
			txt_path = os.path.join(txt_dir, txt_file)
			# txt
			# some wav files dont have correspond txt file
			try:
				with open(txt_path, 'r') as f:
					text = f.read()
				text = re.sub('\n', '', text)
			except:
				continue
			# write
			futures.append(executor.submit(partial(_process_utterance, mel_dir, linear_dir, wav_dir, basename, wav_path, text, hparams, ct)))
		ct += 1
	return [future.result() for future in tqdm(futures) if future.result() is not None]


def build_from_path_THCHS(hparams, input_dir, mel_dir, linear_dir, wav_dir, n_jobs=12, tqdm=lambda x: x):
	"""
	Preprocesses the speech dataset from a gven input path to given output directories

	Args:
		- hparams: hyper parameters
		- input_dir: input directory that contains the files to prerocess
		- mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
		- linear_dir: output directory of the preprocessed speech linear-spectrogram dataset
		- wav_dir: output directory of the preprocessed speech audio dataset
		- n_jobs: Optional, number of worker process to parallelize across
		- tqdm: Optional, provides a nice progress bar

	Returns:
		- A list of tuple describing the train examples. this should be written to train.txt
	"""

	# We use ProcessPoolExecutor to parallelize across processes, this is just for
	# optimization purposes and it can be omited

	executor = ProcessPoolExecutor(max_workers=n_jobs)
	futures = []

	pattern = '([A-Z0-9]+)\\_([0-9]+)\\.(wav)'
	input_path = os.path.join(input_dir, 'data_thchs30/data_thchs30/data')
	data_name = 'THCHS'
	id_dict = {}
	ct = 1

	female_highquality_id_list = ['A2','A4','A8','A11','A12','A13','A14','A19','A22','A23','A32','A34','B2','B4','B7',
								  'B11','B12','B15','B22','B31','B32','C2','C4','C7','C12','C13','C17','C18','C19',
								  'C20','C21','C22','C23','C31','C32','D4','D6','D7','D11','D12','D13','D21','D31']
	female_low_quality_id_list = ['A6','A7','A36','B6','B33','C6','C14','D32']
	male_id_list = ['A5','A9','A33','A35','B8','B21','B34','C8','D8']

	for temp_filename in os.listdir(input_path):
		filename, extension = os.path.splitext(temp_filename)
		if not extension == '.wav':
			continue
		basename = filename
		id = re.findall(pattern, temp_filename)[0][0]  # re.findall(pattern, wav_filename) = [('A7', '157', 'wav.trn')]
		if not id in female_highquality_id_list:
			continue
		if not id in id_dict.keys():
			id_dict[id] = ct
			ct += 1
		wav_path = os.path.join(input_path, filename + '.wav')
		trn_path = os.path.join(input_path, filename + '.wav.trn')
		with open(trn_path, 'r') as f1:
			text = f1.readline().strip()  # '时来运转 遇上 眼前 这位 知音 姑娘 还 因 工程 吃紧 屡 推 婚期\n'
			#phone1 = f1.readline()  # 'zong3 er2 yan2 zhi1 wu2 lun4 na2 li3 ren2 chi1 yi4 wan3 she2 he2 mao1 huo4 zhe3 wa1 he2 shan4 yu2 yu2 xing4 fu2 de5 jia1 ting2 shi4 jue2 bu2 hui4 you3 sun3 shang1 de5\n'
			#phone2 = f1.readline()  # 'z ong3 ee er2 ii ian2 zh ix1 uu u2 l un4 n a2 l i3 r en2 ch ix1 ii i4 uu uan3 sh e2 h e2 m ao1 h uo4 zh e3 uu ua1 h e2 sh an4 vv v2 vv v2 x ing4 f u2 d e5 j ia1 t ing2 sh ix4 j ve2 b u2 h ui4 ii iu3 s un3 sh ang1 d e5\n\n'
		# text2 = re.sub(' ', '', text.strip())  # '时来运转遇上眼前这位知音姑娘还因工程吃紧屡推婚期\n
		futures.append(executor.submit(partial(_process_utterance, mel_dir, linear_dir, wav_dir, basename, wav_path, text, hparams, id_dict[id])))

	return [future.result() for future in tqdm(futures) if future.result() is not None]




def _process_utterance(mel_dir, linear_dir, wav_dir, index, wav_path, text, hparams, speaker_id):
	"""
	Preprocesses a single utterance wav/text pair

	this writes the mel scale spectogram to disk and return a tuple to write
	to the train.txt file

	Args:
		- mel_dir: the directory to write the mel spectograms into
		- linear_dir: the directory to write the linear spectrograms into
		- wav_dir: the directory to write the preprocessed wav into
		- index: the numeric index to use in the spectogram filename
		- wav_path: path to the audio file containing the speech input
		- text: text spoken in the input audio file
		- hparams: hyper parameters

	Returns:
		- A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text)
	"""
	try:
		# Load the audio as numpy array
		wav = audio.load_wav(wav_path, sr=hparams.sample_rate)
	except FileNotFoundError: #catch missing wav exception
		print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
			wav_path))
		return None

	#rescale wav
	if hparams.rescale:
		wav = wav / np.abs(wav).max() * hparams.rescaling_max

	#M-AILABS extra silence specific
	if hparams.trim_silence:
		wav = audio.trim_silence(wav, hparams)

	#Mu-law quantize
	if is_mulaw_quantize(hparams.input_type):
		#[0, quantize_channels)
		out = mulaw_quantize(wav, hparams.quantize_channels)

		#Trim silences
		start, end = audio.start_and_end_indices(out, hparams.silence_threshold)
		wav = wav[start: end]
		out = out[start: end]

		constant_values = mulaw_quantize(0, hparams.quantize_channels)
		out_dtype = np.int16

	elif is_mulaw(hparams.input_type):
		#[-1, 1]
		out = mulaw(wav, hparams.quantize_channels)
		constant_values = mulaw(0., hparams.quantize_channels)
		out_dtype = np.float32

	else:
		#[-1, 1]
		out = wav
		constant_values = 0.
		out_dtype = np.float32

	# Compute the mel scale spectrogram from the wav
	mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
	mel_frames = mel_spectrogram.shape[1]

	if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
		return None

	#Compute the linear scale spectrogram from the wav
	linear_spectrogram = audio.linearspectrogram(wav, hparams).astype(np.float32)
	linear_frames = linear_spectrogram.shape[1]

	#sanity check
	assert linear_frames == mel_frames

	#Ensure time resolution adjustement between audio and mel-spectrogram
	fft_size = hparams.n_fft if hparams.win_size is None else hparams.win_size
	l, r = audio.pad_lr(wav, fft_size, audio.get_hop_size(hparams))

	#Zero pad for quantized signal
	out = np.pad(out, (l, r), mode='constant', constant_values=constant_values)
	assert len(out) >= mel_frames * audio.get_hop_size(hparams)

	#time resolution adjustement
	#ensure length of raw audio is multiple of hop size so that we can use
	#transposed convolution to upsample
	out = out[:mel_frames * audio.get_hop_size(hparams)]
	assert len(out) % audio.get_hop_size(hparams) == 0
	time_steps = len(out)

	# Write the spectrogram and audio to disk
	audio_filename = 'audio-{}.npy'.format(index)
	mel_filename = 'mel-{}.npy'.format(index)
	linear_filename = 'linear-{}.npy'.format(index)
	np.save(os.path.join(wav_dir, audio_filename), out.astype(out_dtype), allow_pickle=False)
	np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
	np.save(os.path.join(linear_dir, linear_filename), linear_spectrogram.T, allow_pickle=False)

	# Return a tuple describing this training example
	return (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, text, speaker_id)
