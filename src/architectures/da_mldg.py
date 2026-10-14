import time
import tensorflow as tf
from .resnet18 import ResNet18

class DA_MLDG(tf.keras.Model):

	def __init__(self, input_shape, num_classes, num_domains, weights=None):
		super(DA_MLDG, self).__init__()
		self.model = ResNet18.model(input_shape=input_shape, num_classes=num_classes, weights=weights)
		self.num_domains = num_domains
		self._input_shape = input_shape
		self._reshape_size = (1,-1,)+tuple(self._input_shape)
		self._num_classes = num_classes
		self._weights = weights
		self.gamma = tf.convert_to_tensor(1.0)	# meta-test
		self.alpha = 1.0 #- self.beta			# meta-train
		self.meta_lr = 0.01						# meta-learn step
		self.create_model = lambda : ResNet18.model(input_shape=self._input_shape, num_classes=self._num_classes, weights=self._weights)
		self.merge_domains = lambda x: (tf.squeeze(tf.reshape(x[0], shape=(1,-1,)+tuple(self._input_shape))), tf.squeeze(tf.reshape(x[1], shape=(1,-1,1))))

	def ood_score(self,model,data,num_samples, temp=1.0):
		logits = model(data,training=False)
		score = tf.math.reduce_logsumexp(logits/temp, axis=-1)
		return tf.reduce_sum(score)/num_samples

	def compile(self, optimizer, loss, metrics):
		super(DA_MLDG, self).compile()
		self.model.compile(loss=loss, metrics=metrics)
		self.optimizer = optimizer
		self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits
		self._metrics = metrics

	def fit(self, data, epochs=1, validation_data=None, validation_freq=1, verbose=0):
		start = time.time()
		for epoch in range(epochs):
			acc, loss, train_loss, test_loss = 0.0 ,0.0, 0.0, 0.0
			progbar = tf.keras.utils.Progbar(target=len(data)-1, width=30, verbose=verbose if verbose != 2 else 0, interval=0.05, stateful_metrics=None, unit_name='step')
			if verbose==1: tf.print('\nEpoch %d/%d' %(epoch+1,epochs))
			for step, (x,y,d) in enumerate(data):
				step_loss, step_train_loss, step_test_loss = self.train_step(data=(x,y,d))
				loss+=step_loss
				train_loss+=step_train_loss
				test_loss+=step_test_loss
				# Plot
				if step < len(data)-1 or not validation_data or len(data)==1:
					progbar.update(current=step, values=[('loss', float(loss/(step+1))), ('train_loss', float(train_loss/(step+1))), ('test_loss', float(test_loss/(step+1)))])

			if (validation_data is not None) and (((epoch+1)%validation_freq) == 0):
				losss,acc = self.model.evaluate(validation_data, verbose=0)
				progbar.update(current=step, values=[('val_loss', float(losss)), ('val_accuracy', float(acc))])

		if verbose==2:
			tf.print('Epoch %d/%d\n%d/%d - %.1fs - train loss: %.4f - val_accuracy: %.4f' %(epoch+1,epochs,step,step, time.time()-start, loss/len(data),float(acc)))

	def copy_model(self):
		copied_model = self.create_model()
		for i in range(len(copied_model.trainable_weights)): 
			copied_model.trainable_weights[i].assign = self.model.trainable_weights[i]
		copied_model.compile(optimizer=self.optimizer)
		return copied_model

	def train_step(self, data):

		(x,y,domain_idxs) = data
		_idx = tf.random.uniform(shape=(), minval=0, maxval=len(data[0]), dtype=tf.dtypes.int64)
		test_domain_idx = tf.gather(domain_idxs,[_idx])
		train_mask, test_mask = tf.not_equal(test_domain_idx, domain_idxs), tf.equal(test_domain_idx, domain_idxs)
		train_data, train_labels = tf.boolean_mask(x, train_mask), tf.boolean_mask(y, train_mask)
		test_data, test_labels = tf.boolean_mask(x, test_mask), tf.boolean_mask(y, test_mask)
		num_train_samples = tf.cast(tf.shape(train_data)[0], dtype=tf.float32)
		num_test_samples = tf.cast(tf.shape(test_data)[0], dtype=tf.float32)
		with tf.GradientTape() as test_tape:
			# Run meta-train step
			with tf.GradientTape() as train_tape:
				logits = self.model(train_data, training=True)
				train_loss = tf.reduce_sum(self.loss(labels=train_labels, logits=logits))/num_train_samples
			gradients = train_tape.gradient(train_loss, self.model.trainable_variables)
			# Temporary copy of model to apply gradients
			temp_model = self.copy_model()
			for i in range(len(self.model.trainable_weights)):
				temp_model.trainable_weights[i].assign = tf.subtract(self.model.trainable_weights[i],tf.multiply(self.meta_lr, gradients[i]))

			# Compute domain shift margin (gamma)
			self.gamma = tf.abs(
							tf.math.subtract(
								self.ood_score(temp_model,train_data,num_train_samples),
								self.ood_score(temp_model,test_data,num_test_samples)
							)
						)
			# Run meta-test step
			test_logits = temp_model(test_data, training=True)
			test_loss = tf.reduce_sum(self.loss(labels=test_labels, logits=test_logits))/num_test_samples
			loss = tf.add(tf.math.scalar_mul(scalar=self.alpha, x=train_loss), tf.math.scalar_mul(scalar=self.gamma, x=test_loss))
		meta_gradients = test_tape.gradient(loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(meta_gradients, self.model.trainable_variables))
		return loss, train_loss, test_loss

	def evaluate(self, *args, **kwargs):
		return self.model.evaluate(*args, **kwargs)

	def predict(self, *args, **kwargs):
		return self.model.predict(*args, **kwargs)

	def save(self, *args, **kwargs):
		return self.model.save(*args, **kwargs)

	def save_weights(self, *args, **kwargs):
		return self.model.save_weights(*args, **kwargs)