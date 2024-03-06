import os
import pickle
import click

from keras import optimizers, losses, metrics, models

from build import builder
from arch import make_model

from utilities.utils import *

@click.group(chain=False, invoke_without_command=True)
@click.option('--debug/--no-debug', help='debug mode flag', default=False)
@click.pass_context
def router_cmd(ctx: click.Context, debug):
    ctx.obj['debug_mode'] = debug 
    invoked_subcommand = ctx.invoked_subcommand 
    if invoked_subcommand is None:
        logger.debug('no subcommand were called')
    else:
        logger.debug(f'{invoked_subcommand} was called')
 
@router_cmd.command()
@click.option('--path2data', help='', type=str, default='data/')
def build(path2data):
    logger.debug('Building...')
    if not os.path.exists(path2data):
        os.makedirs(path2data)
        
    builder(path2data)
       
@router_cmd.command()
@click.option('--path2data', help='path to source data', type=str, default='data/handsigns.pkl/')
@click.option('--path2models', help='path to models', type=click.Path(True), default='models/')
@click.option('--nb_epochs', help='number of epochs', type=int, default=10)
@click.option('--bt_size', help='batch size', type=int, default=16)
def learn(path2data, path2models, nb_epochs, bt_size):
    logger.debug('Learning...')
    if not os.path.exists(path2models):
        os.makedirs(path2models)
        
    training_data = pickle.load(open(path2data, 'rb'))
    cnn_input = [ item['image'] for item in training_data ]
    dnn_input = [ item['matrix'] for item in training_data ]
    label = [ item['label'] for item in training_data ]
    
    batch_cnn_input = np.expand_dims(np.stack(cnn_input), axis=-1) / 255.0
    batch_dnn_input = np.vstack(dnn_input)
    
    sparse_mapper = create_sparse_mapper(label)
    sparsed_label = list(map(lambda e: sparse_mapper[e], label))
    prepared_label = np.vstack(sparsed_label)
    print(prepared_label)
      
    model = make_model(batch_cnn_input.shape[1:], batch_dnn_input.shape[1:], len(np.unique(label)))
    model.summary()
    
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy(name='accuracy')]
    )
    
    model.fit(
        x = {
            'cnn_input': batch_cnn_input,
            'dnn_input': batch_dnn_input
        },
        y=prepared_label,
        epochs=nb_epochs,
        batch_size=bt_size,
        verbose=1,
        shuffle=True
    )
    
    model.save(os.path.join(path2models, 'multimodal-agent.h5'))
    
@router_cmd.command()
@click.option('--path2models', help='path to models', type=click.Path(True), default='models/')
@click.option('--model_name', help='model name', type=str, default='multimodal-agent.h5')
def predict(path2models):
    logger.debug('Predicting...')
    path2model = os.path.join(path2models, 'multimodal-agent.h5')
    predictor = models.load_model(path2model, compile=False)
    predict(predictor)

if __name__ == '__main__':
    router_cmd(obj={})