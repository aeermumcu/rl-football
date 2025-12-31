import json

input_file = 'trained.json'
output_file = 'trained_smart.json'

print(f"Reading {input_file}...")
with open(input_file, 'r') as f:
    data = json.load(f)

keys_to_fix = ['blipAgent', 'bloopAgent', 'blip', 'bloop']

for key in keys_to_fix:
    if key not in data:
        continue
    
    agent_data = data[key]
    weights = agent_data['weights'] # List of Dicts
    
    print(f"Fixing {key}...")
    
    # Block 1: Shared + VH (First 8 layers)
    # Shapes: [12,256], [256], [256,256], [256], [256,128], [128], [128,64], [64]
    # We assume first 8 are always correct (topology start).
    block_base = weights[:8]
    
    remainder = weights[8:]
    
    # Helper to find pair by shape
    # We look for Kernel (matrix). Bias follows it.
    def find_pair(shape_kernel, shape_bias):
        found = []
        for i, w in enumerate(remainder):
            if w['shape'] == shape_kernel:
                # Check next for bias
                if i+1 < len(remainder) and remainder[i+1]['shape'] == shape_bias:
                    return [w, remainder[i+1]]
        return None

    # Target Order on VM (Observed): [S, AH, AO, VH, VO]
    # S: 0-5
    # AH: 6-7 (dense_4)
    # AO: 8-9 (advantage)
    # VH: 10-11 (dense_3)
    # VO: 12-13 (value)
    
    ah_pair = find_pair([128, 64], [64])
    vo_pair = find_pair([64, 1], [1])
    ao_pair = find_pair([64, 10], [10])
    # VH is [128, 64], [64]. Wait.
    # We assumed VH was in block_base.
    # Block base is 8 layers. 
    # [12,256], [256], [256,256], [256], [256,128], [128], [128,64], [64].
    # The last pair in block_base [128,64] IS VH (dense_2 -> x -> VH?) NO.
    # Shared: 256->256->128.
    # The last shared is 128 output. (256, 128).
    # Inputs to VH/AH is 128.
    # So VH is [128, 64]. AH is [128, 64].
    # Both are [128, 64].
    
    # PROBLEM: How to distinguish VH and AH in the file if shapes are identical?
    # In trained.json (Phase 2 Output), we have [S, VH, AH, VO, AO].
    # S ends at index 5.
    # Index 6-7 is VH.
    # Index 8-9 is AH.
    # Both are [128, 64].
    
    # If we take weights[6:8] as VH.
    # And weights[8:10] as AH.
    # We want Target: [S(0-5), AH, AO, VH, VO].
    
    # Re-verify S length.
    # dense (256): 0-1.
    # dense_1 (256): 2-3.
    # dense_2 (128): 4-5.
    # End of Shared.
    
    # So remainder starts at 6.
    # weights[6:] contains 8 tensors: VH, AH, VO, AO.
    block_base = weights[:6] # Only Shared!
    remainder = weights[6:]
    
    # In Phase 2 Output (trained.json), order is [VH, AH, VO, AO].
    # VH matches AH shape.
    # We assume first occurrence is VH, second is AH (Source order).
    
    # Find all [128,64] pairs
    pairs_128_64 = []
    i = 0
    while i < len(remainder) - 1:
        if remainder[i]['shape'] == [128, 64]:
             pairs_128_64.append([remainder[i], remainder[i+1]])
             i += 2
        else:
             i += 1
             
    if len(pairs_128_64) < 2:
        print("Error: Could not find 2 pairs of [128,64]")
        continue
        
    vh_pair = pairs_128_64[0] # First one found (Source: VH)
    ah_pair = pairs_128_64[1] # Second one found (Source: AH)
    
    vo_pair = find_pair([64, 1], [1])
    ao_pair = find_pair([64, 10], [10])
    
    print("Found all blocks. Reassembling for VM: [S, AH, AO, VH, VO]...")
    # NOTE: Swapping VH and AH logic if needed. 
    # If source was [VH, AH], and we want [AH, AO, VH, VO].
    # We use ah_pair, then ao_pair, then vh_pair, then vo_pair.
    
    new_weights = block_base + ah_pair + ao_pair + vh_pair + vo_pair
    
    agent_data['weights'] = new_weights
    
print(f"Saving to {output_file}...")
with open(output_file, 'w') as f:
    json.dump(data, f)
print("Done!")
# Exit loop to avoid executing original file content
exit()
