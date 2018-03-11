# -*- coding: utf-8 -*-
# author: ronniecao
# time: 2018/01/12
# description: util tools for layer
import os


def cal_feel_field(layer):
    y_field, x_field = 1, 1
    
    if layer.ltype == 'conv':
        y_field = min(layer.output_shape[0], y_field + int((layer.y_size+1)/2))
        x_field = min(layer.output_shape[1], x_field + int((layer.x_size+1)/2))
    elif layer.ltype == 'pool':
        y_field = min(layer.output_shape[0], y_field * int(layer.y_size))
        x_field = min(layer.output_shape[1], x_field * int(layer.x_size))
    
    while layer.prev_layer:
        if layer.ltype == 'conv':
            y_field = min(layer.output_shape[0], y_field + int((layer.y_size+1)/2))
            x_field = min(layer.output_shape[1], x_field + int((layer.x_size+1)/2))
        elif layer.ltype == 'pool':
            y_field = min(layer.output_shape[0], y_field * int(layer.y_size))
            x_field = min(layer.output_shape[1], x_field * int(layer.x_size))
        layer = layer.prev_layer

    return [int(y_field), int(x_field)]
