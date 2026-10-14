import tensorflow as tf

class ResNet18:

	@staticmethod
	def block(x, filters, kernel_size=3, stride=1, conv_shortcut=True, bn_axis='channels_last', name=None):
		bn_axis = 3 if bn_axis=='channels_last' else 1
		if conv_shortcut:
			shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, use_bias=False, name=name + '_0_conv')(x)
			shortcut = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
		else:
			shortcut = x
		x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_1_pad')(x)
		x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, use_bias=False, name=name + '_1_conv')(x)
		x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
		x = tf.keras.layers.Activation('relu', name=name + '_1_relu')(x)
		x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
		x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, use_bias=False, name=name + '_2_conv')(x)
		x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
		x = tf.keras.layers.Add(name=name + '_add')([shortcut, x])
		x = tf.keras.layers.Activation('relu', name=name + '_out')(x)
		return x

	@staticmethod
	def stack(x, filters, blocks, stride1=2, conv_shortcut=True, name=None):
		x = __class__.block(x, filters, stride=stride1, conv_shortcut=conv_shortcut, name=name + '_block1')
		for i in range(2, blocks + 1):
			x = __class__.block(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
		return x

	@staticmethod
	def stack_fn(x):
		x = __class__.stack(x, 64, 2, stride1=1, conv_shortcut=False, name='conv2')
		x = __class__.stack(x, 128, 2, name='conv3')
		x = __class__.stack(x, 256, 2, name='conv4')
		return __class__.stack(x, 512, 2, name='conv5')

	@staticmethod
	def model(input_shape=(None,None,3), num_classes=5, use_bias=False, bn_axis='channels_last', use_rsc=False, weights=None):
		inputs = tf.keras.layers.Input(shape=input_shape)
		bn_axis = 3 if bn_axis=='channels_last' else 1
		x = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(inputs)
		x = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)
		x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
		x = tf.keras.layers.Activation('relu', name='conv1_relu')(x)
		x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
		x = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)
		x = __class__.stack_fn(x)

		if use_rsc:
			model = tf.keras.Model(inputs, x, name='ResNet18')
		else:
			x =  tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
			outputs = tf.keras.layers.Dense(units=num_classes, name='predictions')(x)
			model = tf.keras.Model(inputs, outputs, name='ResNet18')

		if weights:
			model.load_weights(weights, by_name=True, skip_mismatch=True)

		return model
