import json

def inspect(filename):
    print(f"--- Inspecting {filename} ---")
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Check blipAgent
        weights = data['blipAgent']['weights']
        for i, w in enumerate(weights):
            print(f"Index {i}: Shape {w['shape']}")
            
    except Exception as e:
        print(f"Error: {e}")

inspect('trained.json')
inspect('trained_fixed.json')
