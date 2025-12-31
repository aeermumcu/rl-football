import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import json
import numpy as np

def create_dueling_dqn(state_size, action_size):
    input_layer = layers.Input(shape=(state_size,))
    x = layers.Dense(256, activation='relu', name='shared_1')(input_layer)
    x = layers.Dense(256, activation='relu', name='shared_2')(x)
    x = layers.Dense(128, activation='relu', name='shared_3')(x)
    
    # Advantage Stream (Defined FIRST to match trained_final.json)
    advantage_stream = layers.Dense(64, activation='relu', name='advantage_hidden')(x)
    advantage = layers.Dense(action_size, activation='linear', name='advantage_out')(advantage_stream)
    
    # Value Stream
    value_stream = layers.Dense(64, activation='relu', name='value_hidden')(x)
    value = layers.Dense(1, activation='linear', name='value_out')(value_stream)
    
    # Combine: Q = V + (A - mean(A))
    mean_a = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1, keepdims=True))(advantage)
    a_centered = layers.Subtract()([advantage, mean_a])
    q = layers.Add()([value, a_centered])
    
    model = models.Model(inputs=input_layer, outputs=q)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='mse')
    return model

print("Building model...")
model = create_dueling_dqn(12, 10)
print("Model built.")

print("Expected attributes:")
for w in model.get_weights():
    print(f"Shape: {w.shape}")

print("-" * 20)
print("Loading weights/trained.json...")

try:
    with open("weights/trained.json", 'r') as f:
        data = json.load(f)
    
    # Try blipAgent
    if 'blipAgent' in data:
        weights = data['blipAgent']['weights']
    else:
        # Maybe old format or scrambled
        # Try finding the list
        weights = data['weights'] if 'weights' in data else []
    
    print(f"File contains {len(weights)} weight tensors.")
    
    np_weights = []
    for i, w in enumerate(weights):
        print(f"Item {i} type: {type(w)}")
        
        if isinstance(w, dict):
            # If TF.js format { 'val': [...], 'shape': ... }
            if 'data' in w:
                 arr = np.array(w['data']).reshape(w['shape'])
            elif 'val' in w:
                 arr = np.array(w['val'])
            else:
                 print(f"Warning: Dict without 'data' or 'val' at index {i}")
                 arr = np.array(w) 
        else:
            # Assume list of numbers
            arr = np.array(w)
            
        print(f"Item {i} array shape: {arr.shape}")
        np_weights.append(arr)
        
    print("Distributing weights by name...")
    
    # File Structure Corrected: [S, AH, VH, AO, VO]
    # S: Indices 0-5
    # AH: 6-7 (128, 64)
    # VH: 8-9 (128, 64)
    # AO: 10-11 (64, 10)
    # VO: 12-13 (64, 1)
    
    # Shared
    model.get_layer('shared_1').set_weights([np_weights[0], np_weights[1]])
    model.get_layer('shared_2').set_weights([np_weights[2], np_weights[3]])
    model.get_layer('shared_3').set_weights([np_weights[4], np_weights[5]])
    
    # Advantage (AH from 6-7, AO from 10-11)
    model.get_layer('advantage_hidden').set_weights([np_weights[6], np_weights[7]])
    model.get_layer('advantage_out').set_weights([np_weights[10], np_weights[11]])
    
    # Value (VH from 8-9, VO from 12-13)
    model.get_layer('value_hidden').set_weights([np_weights[8], np_weights[9]])
    model.get_layer('value_out').set_weights([np_weights[12], np_weights[13]])
    
    print("SUCCESS: Weights loaded by name.")
    
    # Save as H5
    h5_path = "weights/trained.weights.h5"
    print(f"Saving to {h5_path}...")
    model.save_weights(h5_path)
    print("SUCCESS: H5 saved.")
    
    # Verification (Optional)
    # model.set_weights(np_weights) # This would crash if topology differs, skipping.
    
except Exception as e:
    print(f"ERROR: {e}")
