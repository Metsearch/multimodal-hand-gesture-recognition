import itertools as it
import cv2

import numpy as np
from scipy.spatial.distance import euclidean as measure

from .log import logger

def create_screen(name, width, height, position=None):
	cv2.namedWindow(name, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(name, width, height)
	if position is not None:
		cv2.moveWindow(name, *position)

def get_nodes(landmark):
	return np.array([ [pnt.x, pnt.y] for pnt in landmark])

def normalize_nodes(nodes, scaler):
	rescaled_nodes = nodes * scaler
	return rescaled_nodes.astype('int32')

def make_adjacency_matrix(nodes):
	nb_nodes = len(nodes)
	pairwise = list(it.product(nodes, nodes))
	weighted_edges = np.array([ measure(*item) for item in pairwise ])
	return np.reshape(weighted_edges, (nb_nodes, nb_nodes)) / np.max(weighted_edges)

def get_contours(nodes):
	return cv2.boundingRect(nodes)

def draw_message_on_screen(target_screen, message, message_config, message_position):
	(tw, th), tb = cv2.getTextSize(message, **message_config)
	tx, ty = message_position
	cv2.putText(target_screen, message, (tx - tw // 2, ty + th // 2 + tb), color=(0, 255, 255), **message_config)

def create_sparse_mapper(labels):
	unique_values = np.unique(labels)
	return dict(zip(unique_values, range(len(unique_values))))

if __name__ == '__main__':
    logger.info('Testing utils...')