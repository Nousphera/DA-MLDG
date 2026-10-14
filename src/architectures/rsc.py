import os
from glob import glob
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from .resnet18 import ResNet18

class RSC():

	def __init__(self, input_shape, num_classes, weights, percentile=66, batch_percentage=33, trainable_backbone=True, **kwargs):
		self.backbone = ResNet18.model(input_shape=input_shape, num_classes=num_classes, use_rsc=True, weights=weights)
		self.classification_head = tf.keras.models.Sequential([tf.keras.layers.Dense(512), tf.keras.layers.Dense(num_classes)])
		self.model = self._build_model(trainable_backbone=trainable_backbone)
		self.compiled = False
		self.percentile = percentile
		self.batch_percentage = batch_percentage
		self.z = None

	def _build_model(self, trainable_backbone=False):
		if not trainable_backbone:
			for layer in self.backbone.layers:
				layer.trainable=False
		inputs = tf.keras.Input(shape=self.backbone.input.shape[1:])
		x = self.backbone(inputs)
		x = tf.keras.layers.GlobalAveragePooling2D()(x)
		outputs = self.classification_head(x)
		return tf.keras.Model(inputs=[inputs], outputs=[outputs], name='rsc_model')

	def _call(self, inputs, mask=None):
		self.z = self.backbone(inputs)
		if mask != None: x = self.z * mask
		else:  x = self.z
		x = tf.keras.layers.GlobalAveragePooling2D()(x)
		return self.classification_head(x)

	def _backup_checkpoint(self,max_bck):
		if os.path.isdir(self.ckp_dir):
			bck_dir = self.ckp_dir + "_old"
			bck_list = [int(s.split('_')[-1][3:]) for s in glob(f"{bck_dir}*")]
			if len(bck_list):
				i = min(max(bck_list)+1,max_bck)
			else:
				i = 0
			bck_dir += f'{i}'
			os.rename(self.ckp_dir,bck_dir)
			print(f'Checkpoint backup saved for model {self.name}: {bck_dir}')
		if os.path.isdir(self.log_dir):
			bck_dir = self.log_dir + "_old"
			bck_list = [int(s.split('_')[-1][3:]) for s in glob(f"{bck_dir}*")]
			if len(bck_list):
				i = min(max(bck_list)+1,max_bck)
			else:
				i = 0
			bck_dir += f'{i}'
			os.rename(self.log_dir,bck_dir)
			print(f'Tensorboard graph backup saved for model {self.name}: {bck_dir}')

	def _batch_dataset_default(self, X, y, batch_size=128, buffer_size=10000):
		ds = tf.data.Dataset.from_tensor_slices((X,y))
		ds = ds.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
		return ds

	def _compute_validation_step(self,validation_freq, steps_per_epoch):
		if validation_freq == "step":
			return 1
		elif validation_freq == "epoch":
			return steps_per_epoch
		elif isinstance(validation_freq,int):
			return int(steps_per_epoch*validation_freq)
		else:
			raise ValueError(f'Wrong "validation_freq": {validation_freq}. Acceptable values are "step", "epoch" or int.')

	def compile(self, optimizer, loss, metrics, pre_process_fc=None,
			checkpoint_dir='checkpoint', log_dir='logs', bin_dir='bin',
			do_not_restore=True, save_old=False, max_bck=3, name='rsc'):

		self.compiled = True
		self.name = name
		self.loss = loss
		self.metrics = metrics
		self.log_dir = os.path.join(log_dir, self.name)
		self.ckp_dir = os.path.join(checkpoint_dir, self.name)
		self.bin_dir = os.path.join(bin_dir, self.name)

		self.train_loss = tf.keras.metrics.Mean(name='train_loss')
		self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
		self.val_loss = tf.keras.metrics.Mean(name='val_loss')
		self.val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')

		self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0), loss=tf.Variable(np.inf), accuracy=tf.Variable(0.), optimizer=optimizer, model=self.model)
		self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,max_to_keep=1,directory=self.ckp_dir)
		self.batch_dataset = pre_process_fc if pre_process_fc!=None else self._batch_dataset_default

		if do_not_restore:
			if save_old:
				self._backup_checkpoint(max_bck)
		else:
			self.restore()

	def fit(self, X=None, epochs=100, batch_size=128, track="acc",
		validation_data=None, validation_freq=1, save_best_only=True, verbose=1):

		track = track.lower()
		if track not in ["acc","loss"]: raise ValueError(f"Cannot track {track}.")

		self.nun_samples = int(batch_size*self.batch_percentage * 0.01)
		train_ds = X
		writer_train = tf.summary.create_file_writer(os.path.join(self.log_dir, f'train'))
		stateful_metrics=['loss', 'acc']
		loss_to_track = tf.constant(np.inf)
		accuracy_to_track = tf.constant(0.)
		total_step = tf.cast(self.checkpoint.step,tf.int64)
		steps_per_epoch = len(train_ds)

		if validation_data is not None:
			val_ds = validation_data
			writer_val = tf.summary.create_file_writer(os.path.join(self.log_dir, f'val'))
			stateful_metrics.append('val_loss')
			stateful_metrics.append('val_acc')
			validation_freq = self._compute_validation_step(validation_freq, steps_per_epoch)
			if verbose!=0: print(f"Running validation every {validation_freq} steps.")

		for epoch in range(epochs):
			print("Epoch {}/{}".format(epoch+1, epochs))
			pb_i = tf.keras.utils.Progbar(steps_per_epoch, stateful_metrics=stateful_metrics)

			for step, (x_train_step, y_train_step) in enumerate(train_ds):
				if step == 0:
					self.train_loss.reset_states()
					self.train_accuracy.reset_states()
				total_step += 1
				self._train_step(x_train_step, y_train_step)
				self.checkpoint.step.assign_add(1)
				with writer_train.as_default():
					tf.summary.scalar('acc', self.train_accuracy.result(), step=total_step)
					tf.summary.scalar('loss', self.train_loss.result(), step=total_step)
				writer_train.flush()
				values = [('loss', self.train_loss.result()), ('acc', self.train_accuracy.result())]

				if validation_data is not None:
					if step!=0 and ((step + 1)%validation_freq)==0:
						self.val_loss.reset_states()
						self.val_accuracy.reset_states()
						for x_val_step, y_val_step in val_ds:
							self._test_step(x_val_step, y_val_step)
						with writer_val.as_default():
							tf.summary.scalar('acc', self.val_accuracy.result(), step=total_step)
							tf.summary.scalar('loss', self.val_loss.result(), step=total_step)
						writer_val.flush()
						values.append(('val_loss', self.val_loss.result()))
						values.append(('val_acc', self.val_accuracy.result()))
						loss_to_track = self.val_loss.result()
						accuracy_to_track = self.val_accuracy.result()

				else: # if validation is not available, track training
					loss_to_track = self.train_loss.result()
					accuracy_to_track = self.train_accuracy.result()

				pb_i.add(1, values=values) #update bar

				if save_best_only:
					if (track=="loss" and loss_to_track >= self.checkpoint.loss) or (track=="acc" and accuracy_to_track <= self.checkpoint.accuracy):
						continue

				self.checkpoint.loss = loss_to_track
				self.checkpoint.accuracy = accuracy_to_track
				self.checkpoint_manager.save()

	@tf.function
	def _train_step(self, x, y_true):
		# Normal training
		x, y = x[self.nun_samples:], y_true[self.nun_samples:]
		with tf.GradientTape() as tape:
			logits = self._call(x)
			logits = tf.nn.softmax(logits, axis=-1)
			loss = self.loss(y, logits)
		gradients = tape.gradient(loss, self.checkpoint.model.trainable_variables)
		self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))
		self.metrics[0].reset_states()
		self.train_loss(loss)
		self.train_accuracy(self.metrics[0](y, logits))

		# Create feature mask
		x_rsc, y_rsc = x[:self.nun_samples], y_true[:self.nun_samples]
		y_rsc_one_hot = tf.one_hot(indices=tf.cast(y_rsc, dtype=tf.int32), depth=tf.cast(self.classification_head.layers[-1].units, dtype=tf.int32), dtype=tf.float32)
		with tf.GradientTape() as tape:
			logits = self._call(x_rsc)
			gz_func = tf.reduce_sum(logits * y_rsc_one_hot, axis=-1)
			choose_sw_cw = np.random.randint(0, 9)

		if choose_sw_cw <= 4:
			gradients = tape.gradient(gz_func, self.z)
			gradients_sw = tf.reduce_mean(gradients, axis=[-1])
			mask = tf.cast(gradients_sw < tfp.stats.percentile(x=gradients_sw, q=self.percentile, axis=[1,2], keepdims=True), dtype=tf.float32)[...,None]
		else:
			gradients = tape.gradient(gz_func, self.z)
			gradients_cw = tf.reduce_mean(gradients, axis=[1,2])
			mask = tf.cast(gradients_cw < tfp.stats.percentile(x=gradients_cw, q=self.percentile, axis=[1], keepdims=True), dtype=tf.float32)
			mask = tf.reshape(mask, (mask.shape[0],1,1,-1))

		# Train with hidden features
		with tf.GradientTape() as tape:
			logits = self._call(x_rsc, mask=mask)
			logits = tf.nn.softmax(logits, axis=-1)
			loss = self.loss(y_rsc, logits)
		gradients = tape.gradient(loss, self.checkpoint.model.trainable_variables)
		self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))

	def restore(self):
		if self.checkpoint_manager.latest_checkpoint and self.compiled:
			self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
			print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')

	def predict(self, x, **kwargs):
		y_pred = []
		for i in x:
			y_pred.extend(tf.nn.softmax(self.model(i), axis=-1))
		return y_pred

	def evaluate(self, X=None, verbose=1, **kwargs):

		if not self.compiled:
			raise NotImplementedError(f'evaluate() method is called, but model not yet compiled!')

		if verbose!=0:
			stateful_metrics=['loss', 'acc']
			pb_i = tf.keras.utils.Progbar(len(X), stateful_metrics=stateful_metrics)

		loss_mean = tf.keras.metrics.Mean(name='loss')
		self.metrics[0].reset_states()
		values = [('loss', loss_mean.result()), ('acc', self.metrics[0].result())]
		for _,(x, y) in enumerate(X):
			logits = tf.nn.softmax(self.model(x, training=False), axis=-1)
			loss_mean(self.loss(y, logits))
			self.metrics[0](y, logits)
			values.append(('loss', loss_mean.result()))
			values.append(('acc', self.metrics[0].result()))
			if verbose!=0: pb_i.add(1, values=values)
		return (loss_mean.result().numpy(), self.metrics[0].result().numpy())

	@tf.function
	def _test_step(self, x, y_true):
		logits = self._call(x)
		loss = self.loss(y_true, logits)
		self.metrics[0].reset_states()
		metric = self.metrics[0](y_true, logits)
		self.val_loss(loss)
		self.val_accuracy(metric)

	def save(self, *args, **kwargs):
		self.model.save(*args, **kwargs)

	def save_weights(self, *args, **kwargs):
		self.model.save_weights(*args, **kwargs)

	def summary(self, *args, **kwargs):
		return self.model.summary(*args, **kwargs)