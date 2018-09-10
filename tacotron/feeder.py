import os, json, random
import threading
import time
import traceback

import numpy as np
import tensorflow as tf
from infolog import log
from sklearn.model_selection import train_test_split
from tacotron.utils.text import text_to_sequence, text_to_sequence2

_batches_per_group = 32

class Feeder:
	"""
		Feeds batches of data into queue on a background thread.
	"""

	def __init__(self, coordinator, metadata_filename_list, hparams):
		super(Feeder, self).__init__()
		self._coord = coordinator
		self._hparams = hparams
		self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
		self._train_offset = 0
		self._test_offset = 0

		if self._hparams.chinese_dict:
			self.text_to_sequence = text_to_sequence2
		else:
			self.text_to_sequence = text_to_sequence

		# Load metadata
		self._metadata = []
		id_num = 0
		for item in metadata_filename_list:
			basedir = os.path.dirname(item)
			_mel_dir = os.path.join(basedir, 'mels')
			_linear_dir = os.path.join(basedir, 'linear')
			_audio_dir = os.path.join(basedir, 'audio')
			crrt_id_num = 0
			print(item)
			with open(item, encoding='utf-8') as f:
				for line in f:
					line = line.strip().split('|')
					line[6] = int(line[6])
					if line[6] > crrt_id_num:
						crrt_id_num = line[6]
					line[6] = line[6] + id_num
					line[0] = os.path.join(_audio_dir, line[0])
					line[1] = os.path.join(_mel_dir, line[1])
					line[2] = os.path.join(_linear_dir, line[2])
					if int(line[4]) < hparams.max_iters:
						self._metadata.append(line)
			id_num += crrt_id_num
		frame_shift_ms = hparams.hop_size / hparams.sample_rate
		hours = sum([int(x[4]) for x in self._metadata]) * frame_shift_ms / (3600)
		log('Loaded metadata for {} examples ({:.2f} hours)'.format(len(self._metadata), hours))
		'''
		#check the metadata
		with open('text.txt', 'w') as f:
			for item in self._metadata:
				f.write(str(item) + '\n')
		'''

		#Train test split
		if hparams.tacotron_test_size is None:
			assert hparams.tacotron_test_batches is not None

		test_size = (hparams.tacotron_test_size if hparams.tacotron_test_size is not None
			else hparams.tacotron_test_batches * hparams.tacotron_batch_size)
		indices = np.arange(len(self._metadata))
		train_indices, test_indices = train_test_split(indices,
			test_size=test_size, random_state=hparams.tacotron_data_random_state)

		#Make sure test_indices is a multiple of batch_size else round up
		len_test_indices = self._round_down(len(test_indices), hparams.tacotron_batch_size)
		extra_test = test_indices[len_test_indices:]
		test_indices = test_indices[:len_test_indices]
		train_indices = np.concatenate([train_indices, extra_test])

		self._train_meta = list(np.array(self._metadata)[train_indices])
		self._test_meta = list(np.array(self._metadata)[test_indices])

		self.test_steps = len(self._test_meta) // hparams.tacotron_batch_size

		if hparams.tacotron_test_size is None:
			assert hparams.tacotron_test_batches == self.test_steps

		#pad input sequences with the <pad_token> 0 ( _ )
		self._pad = 0
		#explicitely setting the padding to a value that doesn't originally exist in the spectogram
		#to avoid any possible conflicts, without affecting the output range of the model too much
		if hparams.symmetric_mels:
			self._target_pad = -(hparams.max_abs_value + .1)
		else:
			self._target_pad = -0.1
		#Mark finished sequences with 1s
		self._token_pad = 1.

		with tf.device('/cpu:0'):
			# Create placeholders for inputs and targets. Don't specify batch size because we want
			# to be able to feed different batch sizes at eval time.
			self._placeholders = [
			tf.placeholder(tf.int32, shape=(None, None), name='inputs'),
			tf.placeholder(tf.int32, shape=(None, ), name='input_lengths'),
			tf.placeholder(tf.float32, shape=(None, None, hparams.num_mels), name='mel_targets'),
			tf.placeholder(tf.float32, shape=(None, None), name='token_targets'),
			tf.placeholder(tf.float32, shape=(None, None, hparams.num_freq), name='linear_targets'),
			tf.placeholder(tf.float32, shape=(None, None), name='wavs'),
			tf.placeholder(tf.int32, shape=(None,), name='identitities'),
			tf.placeholder(tf.int32, shape=(None, ), name='targets_lengths'),
			]

			# Create queue for buffering data
			queue = tf.FIFOQueue(8, [tf.int32, tf.int32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int32, tf.int32], name='input_queue')
			self._enqueue_op = queue.enqueue(self._placeholders)
			self.inputs, self.input_lengths, self.mel_targets, self.token_targets, self.linear_targets, self.wavs, self.identities, self.targets_lengths = queue.dequeue()

			self.inputs.set_shape(self._placeholders[0].shape)
			self.input_lengths.set_shape(self._placeholders[1].shape)
			self.mel_targets.set_shape(self._placeholders[2].shape)
			self.token_targets.set_shape(self._placeholders[3].shape)
			self.linear_targets.set_shape(self._placeholders[4].shape)
			self.wavs.set_shape(self._placeholders[5].shape)
			self.identities.set_shape(self._placeholders[6].shape)
			self.targets_lengths.set_shape(self._placeholders[7].shape)

			# Create eval queue for buffering eval data
			eval_queue = tf.FIFOQueue(1, [tf.int32, tf.int32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int32, tf.int32], name='eval_queue')
			self._eval_enqueue_op = eval_queue.enqueue(self._placeholders)
			self.eval_inputs, self.eval_input_lengths, self.eval_mel_targets, self.eval_token_targets, self.eval_linear_targets, self.eval_wavs, self.eval_identities, self.eval_targets_lengths = eval_queue.dequeue()

			self.eval_inputs.set_shape(self._placeholders[0].shape)
			self.eval_input_lengths.set_shape(self._placeholders[1].shape)
			self.eval_mel_targets.set_shape(self._placeholders[2].shape)
			self.eval_token_targets.set_shape(self._placeholders[3].shape)
			self.eval_linear_targets.set_shape(self._placeholders[4].shape)
			self.eval_wavs.set_shape(self._placeholders[5].shape)
			self.eval_identities.set_shape(self._placeholders[6].shape)
			self.eval_targets_lengths.set_shape(self._placeholders[7].shape)

		# Load phone dict: If enabled, this will randomly substitute some words in the training data with
		# their ARPABet equivalents, which will allow you to also pass ARPABet to the model for
		# synthesis (useful for proper nouns, etc.)
		if hparams.per_cen_phone_input:
			char_2_phone_dict_path = './tacotron/utils/symbols/char_2_phone_dict.json'
			if not os.path.isfile(char_2_phone_dict_path):
				raise Exception('no char_2_phone dict found')
			with open(char_2_phone_dict_path, 'r') as f:
				self._phone_dict = json.load(f)
				log('Loaded characters to phones dict from %s' % char_2_phone_dict_path)
		else:
			self._phone_dict = None


	def start_threads(self, session):
		self._session = session
		thread = threading.Thread(name='background', target=self._enqueue_next_train_group)
		thread.daemon = True #Thread will close when parent quits
		thread.start()

		thread = threading.Thread(name='background', target=self._enqueue_next_test_group)
		thread.daemon = True #Thread will close when parent quits
		thread.start()

	def _get_test_groups(self):
		meta = self._test_meta[self._test_offset]
		self._test_offset += 1

		text = meta[5]


		input_data = np.asarray(self.text_to_sequence(text, self._cleaner_names), dtype=np.int32)
		mel_target = np.load(meta[1])
		#Create parallel sequences containing zeros to represent a non finished sequence
		token_target = np.asarray([0.] * (len(mel_target) - 1))
		linear_target = np.load(meta[2])
		wav_target = np.load(meta[0])
		identity = int(meta[6])
		return (input_data, mel_target, token_target, linear_target, wav_target, identity, len(mel_target))

	def make_test_batches(self):
		start = time.time()

		# Read a group of examples
		n = self._hparams.tacotron_batch_size
		r = self._hparams.outputs_per_step

		#Test on entire test set
		examples = [self._get_test_groups() for i in range(len(self._test_meta))]

		# Bucket examples based on similar output sequence length for efficiency
		examples.sort(key=lambda x: x[-1])
		batches = [examples[i: i+n] for i in range(0, len(examples), n)]
		np.random.shuffle(batches)

		log('\nGenerated {} test batches of size {} in {:.3f} sec'.format(len(batches), n, time.time() - start))
		return batches, r

	def _enqueue_next_train_group(self):
		while not self._coord.should_stop():
			start = time.time()

			# Read a group of examples
			n = self._hparams.tacotron_batch_size
			r = self._hparams.outputs_per_step
			examples = [self._get_next_example() for i in range(n * _batches_per_group)]

			# Bucket examples based on similar output sequence length for efficiency
			examples.sort(key=lambda x: x[-1])
			batches = [examples[i: i+n] for i in range(0, len(examples), n)]
			np.random.shuffle(batches)

			log('\nGenerated {} train batches of size {} in {:.3f} sec'.format(len(batches), n, time.time() - start))
			for batch in batches:
				feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch, r)))
				self._session.run(self._enqueue_op, feed_dict=feed_dict)

	def _enqueue_next_test_group(self):
		#Create test batches once and evaluate on them for all test steps
		test_batches, r = self.make_test_batches()
		while not self._coord.should_stop():
			for batch in test_batches:
				feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch, r)))
				self._session.run(self._eval_enqueue_op, feed_dict=feed_dict)

	def _get_next_example(self):
		"""Gets a single example (input, mel_target, token_target, linear_target, mel_length) from_ disk
		"""
		if self._train_offset >= len(self._train_meta):
			self._train_offset = 0
			np.random.shuffle(self._train_meta)

		meta = self._train_meta[self._train_offset]
		self._train_offset += 1

		text = meta[5]
		if self._phone_dict:
			self._p_phone_sub = random.random() - 0.5 + (self._hparams.per_cen_phone_input * 2 - 0.5)
			text2 = ''
			for word in text.split(' '):
				exist_alpha = False
				for item in word:
					if is_alphabet(item):
						exist_alpha = True
						break
				phone = self._maybe_get_arpabet(word)
				if not text2 and exist_alpha:
					text2 = text2 + ' '
				text2 += phone
			text = text2
		input_data = np.asarray(self.text_to_sequence(text, self._cleaner_names), dtype=np.int32)
		mel_target = np.load(meta[1])
		#Create parallel sequences containing zeros to represent a non finished sequence
		token_target = np.asarray([0.] * (len(mel_target) - 1))
		linear_target = np.load(meta[2])
		wav_target = np.load(meta[0])
		identity = int(meta[6])
		return (input_data, mel_target, token_target, linear_target, wav_target, identity, len(mel_target))


	def _maybe_get_arpabet(self, word):
		try:
			phone = self._phone_dict[word]
			phone = ' '.join(phone)
		except:
			phone = None
		# log('%s is not found in the char 2 phone dict' % word)
		return '{%s}' % phone if phone is not None and random.random() < self._p_phone_sub else word


	def _prepare_batch(self, batch, outputs_per_step):
		np.random.shuffle(batch)
		inputs = self._prepare_inputs([x[0] for x in batch])
		input_lengths = np.asarray([len(x[0]) for x in batch], dtype=np.int32)
		mel_targets = self._prepare_targets([x[1] for x in batch], outputs_per_step)
		#Pad sequences with 1 to infer that the sequence is done
		token_targets = self._prepare_token_targets([x[2] for x in batch], outputs_per_step)
		linear_targets = self._prepare_targets([x[3] for x in batch], outputs_per_step)
		wavs = self._prepare_inputs([x[4] for x in batch])
		identities = np.asarray([x[5] for x in batch], dtype=np.int32)
		targets_lengths = np.asarray([x[-1] for x in batch], dtype=np.int32) #Used to mask loss
		return (inputs, input_lengths, mel_targets, token_targets, linear_targets, wavs, identities, targets_lengths)

	def _prepare_inputs(self, inputs):
		max_len = max([len(x) for x in inputs])
		return np.stack([self._pad_input(x, max_len) for x in inputs])

	def _prepare_targets(self, targets, alignment):
		max_len = max([len(t) for t in targets])
		return np.stack([self._pad_target(t, self._round_up(max_len, alignment)) for t in targets])

	def _prepare_token_targets(self, targets, alignment):
		max_len = max([len(t) for t in targets]) + 1
		return np.stack([self._pad_token_target(t, self._round_up(max_len, alignment)) for t in targets])

	def _pad_input(self, x, length):
		return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=self._pad)

	def _pad_target(self, t, length):
		return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=self._target_pad)

	def _pad_token_target(self, t, length):
		return np.pad(t, (0, length - t.shape[0]), mode='constant', constant_values=self._token_pad)

	def _round_up(self, x, multiple):
		remainder = x % multiple
		return x if remainder == 0 else x + multiple - remainder

	def _round_down(self, x, multiple):
		remainder = x % multiple
		return x if remainder == 0 else x - remainder

def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
        return True
    else:
        return False
