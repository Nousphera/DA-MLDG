import os
import itertools
import functools
import tensorflow as tf

lr_schedule = lambda lr: tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=10000, decay_rate=0.96, staircase=False)
opt_fn = lambda x: tf.keras.optimizers.SGD(learning_rate=lr_schedule(x[1]), momentum=0.85) if x[0]!='rsc' else tf.keras.optimizers.Adam(lr_schedule(x[1]))
loss_fn = lambda x: tf.keras.losses.SparseCategoricalCrossentropy(from_logits=x[0], name=x[1])
metric = lambda n: tf.keras.metrics.SparseCategoricalAccuracy(name=n)

SCENARIOS = {
	'pacs': ['art_painting','cartoon','photo','sketch'],
	'eng4': ['iemocap','ravdess','savee','crema_d'],
	'scn1': ['ravdess','savee','crema_d', 'iemocap',],
	'scn2': ['crema_d', 'savee', 'tess', 'ravdess'],
	'scn3': ['crema_d', 'savee', 'ravdess', 'tess'],
	'scn4': ['savee', 'ravdess', 'emodb', 'cafe'],
	'scn5': ['aesdd', 'emovo', 'shemo', 'emodb'],
	'scn6': ['crema_d', 'savee', 'ravdess', 'shemo'],
	'scn7': ['tess', 'emodb', 'shemo'],
	'all':  ['emodb', 'crema_d', 'cafe', 'emovo', 'shemo', 'tess', 'aesdd', 'savee', 'iemocap', 'ravdess'],
}

class Datasets:

	@staticmethod
	def load_image_datasets(name, fp='./datasets/', image_size=(224,224)):
		ds = {}

		for idx,n in enumerate(SCENARIOS[name]):
			ds[n] = ImageUtils.read_dataset(n,fp=os.path.join(fp,name,n), image_size=image_size, domain_idx=idx)
		num_samples = [len(x) for x in ds.values()]
		num_domains = len(SCENARIOS[name])
		num_classes = 7
		return ds, num_classes, num_samples, num_domains

	@staticmethod
	def load_audio_dataset(name, fp='./datasets/'):
		ds = AudioUtils.read_dataset(name,fp=os.path.join(fp,name), use_common_emo=True, domain_idx=1)
		ds = ds.map(AudioUtils.drop_domain_info, num_parallel_calls=tf.data.AUTOTUNE)
		num_classes = 5
		return ds, num_classes

	@staticmethod
	def load_audio_datasets(name, fp='./datasets/'):
		ds = {}
		for idx,n in enumerate(SCENARIOS[name]):
			ds[n] = AudioUtils.read_dataset(n,fp=os.path.join(fp,n), use_common_emo=True, domain_idx=idx)
		num_samples = [len(x) for x in ds.values()]
		num_domains = len(SCENARIOS[name])
		num_classes = 5

		return ds, num_classes, num_samples, num_domains

	@staticmethod
	def combine_domains(domains):
		ds = tf.data.Dataset.from_tensor_slices(domains)\
			.interleave(lambda x: x, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE)\
				.shuffle(buffer_size=10000, seed=0, reshuffle_each_iteration=True)
		ds = ds.apply(tf.data.experimental.assert_cardinality(sum([len(x) for x in domains])))
		return ds

	@staticmethod
	def create_domain_splits(domains, num_domains=4):
		_mapper = {i:k for i,k in enumerate(list(domains.keys()))}
		_apply_map = lambda i: list(map(lambda x: tuple(map(lambda _x: _mapper[_x],x)),i))
		train_combs = list(itertools.combinations(range(num_domains),r=num_domains-1))
		test_combs = [tuple(set(range(num_domains)) - set(comb)) for comb in train_combs]
		return _apply_map(train_combs), _apply_map(test_combs)

class AudioUtils:

	COMMON_EMOTIONS = 5

	CLASS_NAMES = {
		'iemocap':	['anger', 'happiness', 'sadness', 'disgust', 'fear', 'neutral', 'surprise','frustration', 'other',],
		'savee':	['anger', 'happiness', 'sadness', 'disgust', 'fear', 'neutral', 'surprise'],
		'crema_d':	['anger', 'happiness', 'sadness', 'disgust', 'fear', 'neutral', 'surprise'],
		'ravdess':	['anger', 'happiness', 'sadness', 'disgust', 'fear', 'neutral', 'surprise', 'calm'],
		'cafe':		['anger', 'happiness', 'sadness', 'disgust', 'fear', 'neutral', 'surprise'],
		'aesdd':	['anger', 'happiness', 'sadness', 'disgust', 'fear', 'neutral', 'surprise'],
		'emodb':	['anger', 'happiness', 'sadness', 'disgust', 'fear', 'neutral', 'surprise', 'boredom'],
		'emovo':	['anger', 'happiness', 'sadness', 'disgust', 'fear', 'neutral', 'surprise'],
		'shemo':	['anger', 'happiness', 'sadness', 'disgust', 'fear', 'neutral', 'surprise'],
		'tess':		['anger', 'happiness', 'sadness', 'disgust', 'fear', 'neutral', 'surprise'],
	}

	@staticmethod
	def inject_len_info(ds):
		ds = ds.apply(tf.data.experimental.assert_cardinality(ds.reduce(0, tf.autograph.experimental.do_not_convert(lambda x,_: x+1)).numpy()))
		return ds

	@staticmethod
	def inject_domain_info(x,y,idx):
		return x,y,idx

	@staticmethod
	def drop_domain_info(x,y,_):
		return x,y

	@staticmethod
	def combine_domains(domains):
		ds = tf.data.Dataset.from_tensor_slices(domains)\
			.interleave(lambda x: x, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE)\
				.shuffle(buffer_size=10000, seed=0, reshuffle_each_iteration=True)
		ds = ds.apply(tf.data.experimental.assert_cardinality(sum([len(x) for x in domains])))
		return ds

	@staticmethod
	def lookup_table_from_dict(dictionary, default_value=-1, dtype=tf.int32):
		keys_tensor = tf.cast(tf.constant(list(dictionary.keys())), dtype=dtype)
		vals_tensor = tf.cast(tf.constant(list(dictionary.values())), dtype=tf.int64)
		init = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
		return tf.lookup.StaticHashTable(init, default_value=default_value)

	@staticmethod
	def filter_ds(x, label, idx, classes=[]):
		classes=tf.constant(classes)
		classes_rule = tf.equal(classes, tf.cast(label, classes.dtype))
		reduced_classes = tf.reduce_sum(tf.cast(classes_rule, tf.float32))
		return tf.greater(reduced_classes, tf.constant(0.))

	@staticmethod
	@tf.autograph.experimental.do_not_convert
	def remap_labels(x,y, table):
		return x, table.lookup(y)

	@staticmethod
	def read_audio(fp, label=None):
		waveform, _ = tf.audio.decode_wav(tf.io.read_file(fp))
		return waveform[Ellipsis, 0], label

	@staticmethod
	def pad(x, sequence_length=16000):
		padding = tf.maximum(sequence_length - tf.shape(x)[0], 0)
		left_pad = padding // 2
		right_pad = padding - left_pad
		return tf.pad(x, paddings=[[left_pad, right_pad]])

	@staticmethod
	def logmelspectogram(x, bins=64, sr=16000, fmin=60.0, fmax=7800.0, fft_length=1024):
		s = tf.signal.stft(x, frame_length=400, frame_step=160, fft_length=fft_length)
		x = tf.abs(s)
		w = tf.signal.linear_to_mel_weight_matrix(bins, tf.shape(s)[-1], sr, fmin, fmax)
		x = tf.tensordot(x, w, 1)
		x.set_shape(x.shape[:-1].concatenate(w.shape[-1:]))
		x = tf.math.log(x+1e-6)
		return x[Ellipsis, tf.newaxis]

	@staticmethod
	def read_dataset(dataset_name, fp, use_common_emo=None, domain_idx=None):
		classes = __class__.CLASS_NAMES[dataset_name]
		ds = tf.keras.utils.audio_dataset_from_directory(
			directory=fp, labels='inferred', label_mode='int', class_names=classes, batch_size=None, shuffle=True)
		if domain_idx is not None:
			ds = ds.map(functools.partial(__class__.inject_domain_info,idx=domain_idx), num_parallel_calls=tf.data.AUTOTUNE)
		if use_common_emo is not None: # Use 5 first emo (same for all datasets)
			ds = ds.filter(functools.partial(__class__.filter_ds, classes=list(range(__class__.COMMON_EMOTIONS))))
		ds = __class__.inject_len_info(ds)
		return ds

	@staticmethod
	def train_prep_example(x, y, idx, seconds=1, bins=64, sr=16000, to_float=False, drop_idx=False):
		x = tf.squeeze(x)
		if to_float:
			x = tf.cast(x, tf.float32) / float(tf.int16.max)
		x = __class__.pad(x, sequence_length=int(sr*seconds))
		x = tf.image.random_crop(x, [int(sr*seconds)])
		x = __class__.logmelspectogram(x, bins=bins, sr=sr)
		return (x,y) if drop_idx else (x,y,idx)

	@staticmethod
	def test_prep_example(x,y, idx, bins=64, sr=16000, to_float=False):
		x = tf.squeeze(x)
		if to_float:
			x = tf.cast(x, tf.float32) / float(tf.int16.max)
		x = __class__.logmelspectogram(x, bins=bins, sr=sr)
		return x,y

	@staticmethod
	def train_prep(ds, domains, bins=64, sr=16000, seconds=2, bs=128, drop_idx=False, to_float=False):
		ds = __class__.combine_domains(domains=[ds[x] for x in domains])
		ds = ds.map(functools.partial(__class__.train_prep_example, seconds=seconds, bins=bins, sr=sr, to_float=to_float, drop_idx=drop_idx))\
			.batch(bs, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
		return ds

	@staticmethod
	def test_prep(ds, domains, bins=64, sr=16000, to_float=False):
		ds = __class__.combine_domains(domains=[ds[x] for x in domains])
		ds = ds.map(functools.partial(__class__.test_prep_example, bins=bins, sr=sr, to_float=to_float))\
			.batch(1).prefetch(tf.data.AUTOTUNE)
		return ds

class ImageUtils:


	CLASS_NAMES = {
		'art_painting':	['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house','person',],
		'photo':	['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house','person',],
		'cartoon':	['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house','person',],
		'sketch':	['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house','person',],
	}

	@staticmethod
	def inject_len_info(ds):
		ds = ds.apply(tf.data.experimental.assert_cardinality(ds.reduce(0, tf.autograph.experimental.do_not_convert(lambda x,_: x+1)).numpy()))
		return ds

	@staticmethod
	def inject_domain_info(x,y,idx):
		return x,y,idx

	@staticmethod
	def combine_domains(domains):
		return tf.data.Dataset.from_tensor_slices(domains)\
			.interleave(lambda x: x, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE)\
				.shuffle(buffer_size=10000, seed=0, reshuffle_each_iteration=True)

	@staticmethod
	def read_dataset(dataset_name, fp, domain_idx=None, image_size=(224, 224)):
		classes = __class__.CLASS_NAMES[dataset_name]
		ds = tf.keras.utils.image_dataset_from_directory(
			directory=fp, labels='inferred', label_mode='int', class_names=classes, batch_size=None, image_size=image_size, shuffle=True)
		if domain_idx is not None:
			ds = ds.map(functools.partial(__class__.inject_domain_info,idx=domain_idx), num_parallel_calls=tf.data.AUTOTUNE)
		ds = __class__.inject_len_info(ds)
		return ds

	def train_prep_example(x,y,idx, drop_idx=False):
		x = tf.subtract(x, tf.constant([123.68, 116.779, 103.939], dtype=tf.float32))
		return (x,y) if drop_idx else (x,y,idx)

	def test_prep_example(x,y,idx):
		x = tf.subtract(x, tf.constant([123.68, 116.779, 103.939], dtype=tf.float32))
		return x,y

	@staticmethod
	def train_prep(ds, domains, bs=128, drop_idx=False):
		ds = __class__.combine_domains(domains=[ds[x] for x in domains])
		ds = ds.map(functools.partial(__class__.train_prep_example, drop_idx=drop_idx))\
			.batch(bs, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
		return ds

	@staticmethod
	def test_prep(ds, domains, bs=128):
		ds = __class__.combine_domains(domains=[ds[x] for x in domains])
		ds = ds.map(__class__.test_prep_example).batch(bs).prefetch(tf.data.AUTOTUNE)
		return ds