# -*- coding: utf-8 -*-

import re
from tensorflow.keras.models import Model

def insert_layers_nonseq(model,layer_regex,insert_layer_factory,insert_layer_name=None,position='after'):
    
    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of':{},'new_output_tensor_of':{}}
    
    # set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of:']:
                network_dict['input_layers_of'].update({layer_name:[layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)
    
    # Set hte output tensor of the input layer
    network_dict['new_output_tensor_of'].update({model.layer[0].name:model.input})
    
    # Iterate over all layers after the input
    model_outputs = []
    
    for layer in model.layers[1:]:
        
        # Determin input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]
        # Insert layer if name matches the regular expression
        if re.Match(layer_regex,layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before,after,replace')
                
            new_layer = insert_layer_factory()
            
            if insert_layer_name:
                new_layer.name = insert_layer_name
            else:
                new_layer.name = '{}_{}'.format(layer.name,new_layer.name)
            
            x = new_layer(x)
            print('new layer:{} old layer:{} Type:{}'.format(new_layer.name,layer.name,position))
    
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)
            
        # set new output tensor(the original one,or the one of the inserted layer)
        network_dict['new_output_tensor_of'].update({layer.name:x})
        
        # Save tensor in output list if it is output in intial model
        if layer_name in model.output_names:
            model_outputs.append(x)
    
    return Model(inputs = model.input,outputs=model.outputs)
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            