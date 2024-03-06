from keras import Input, models, layers

from utilities.utils import *

def make_model(cnn_shape, mlp_shape, nb_classes):
	cnn_input = Input(shape=cnn_shape, name='cnn_input')
	dnn_input = Input(shape=mlp_shape, name='mlp_input')

	dnn_hidden_0 = layers.Dense(units=128, activation='relu')(dnn_input)
	dnn_hidden_1 = layers.Dense(units=64, activation='relu')(dnn_hidden_0)

	cnn_hidden_0 = layers.Conv2D(filters=8, kernel_size=(3, 3))(cnn_input)
	cnn_hidden_1 = layers.MaxPooling2D((3, 3))(cnn_hidden_0)
	cnn_hidden_2 = layers.Conv2D(filters=16, kernel_size=(3, 3))(cnn_hidden_1)
	cnn_hidden_3 = layers.MaxPooling2D((3, 3))(cnn_hidden_2)
	cnn_hidden_4 = layers.Flatten()(cnn_hidden_3)

	concatanator = layers.Concatenate()([dnn_hidden_1, cnn_hidden_4])
	normalizer = layers.BatchNormalization()(concatanator)

	fcn_hidden_0 = layers.Dense(units=64, activation='relu')(normalizer)
	fcn_hidden_1 = layers.Dropout(0.5)(fcn_hidden_0)
	fcn_hidden_2 = layers.Dense(units=32, activation='relu')(fcn_hidden_1)
	fcn_hidden_3 = layers.Dropout(0.1)(fcn_hidden_2)
	fcn_hidden_4 = layers.Dense(units=nb_classes, activation='softmax')(fcn_hidden_3)

	return models.Model(
		inputs=[cnn_input, dnn_input],
		outputs=fcn_hidden_4
	)
 
if __name__ == '__main__':
    logger.info('Testing arch...')