import json
import shutil

input_file = 'trained.json'
output_file = 'trained_python.json'

print(f"Reading {input_file}...")
with open(input_file, 'r') as f:
    data = json.load(f)

# The file follows the structure of 'export_weights' in fast_trainer.py
# It has versions for 'blipAgent', 'bloopAgent', etc.
# But they likely point to the same weight data objects in memory?
# In JSON, they are duplicated data. We must fix ALL of them.

keys_to_fix = ['blipAgent', 'bloopAgent', 'blip', 'bloop']

for key in keys_to_fix:
    if key not in data:
        continue
    
    agent_data = data[key]
    weights = agent_data['weights']
    
    if len(weights) != 14:
        print(f"Skipping {key}: Expected 14 weights, found {len(weights)}")
        continue
        
    print(f"Fixing weights for {key}...")
    
    # Source (Phase 2 Export / TF.js Ready):
    # [Shared, VH, AH, VO, AO]
    # Indices:
    # 0-7: S + VH
    # 8-9: AH
    # 10-11: VO
    # 12-13: AO
    
    # Target (Python Keras Standard):
    # [Shared, VH, VO, AH, AO]
    
    # We need to swap AH (8-9) and VO (10-11).
    
    fixed_weights = (
        weights[:8] +      # Shared + VH (Keep)
        weights[10:12] +   # VO (Move from 10-11 to 8-9)
        weights[8:10] +    # AH (Move from 8-9 to 10-11)
        weights[12:]       # AO (Keep)
    )
    
    # Result: [S, VH, VO, AH, AO]
    
    agent_data['weights'] = fixed_weights

print(f"Saving to {output_file}...")
with open(output_file, 'w') as f:
    json.dump(data, f)

print("Done! Try loading 'trained_fixed.json'.")
