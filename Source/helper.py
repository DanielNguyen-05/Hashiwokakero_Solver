import os

# Load grid from file txt
def load_input(filename):
    filepath = os.path.join("Source/Inputs", filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r') as f:
        return [[int(x.strip()) for x in line.strip().split(',')] for line in f]

# Upload the result to file txt  
def upload_output(grid, filename, method):
    output_dir = os.path.join("Source/Outputs", method)
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, filename)

    with open(full_path, 'w', encoding='utf-8') as f:
        for row in grid:
            line = '[ ' + ' , '.join(f'"{cell}"' for cell in row) + ' ]'
            f.write(line + '\n')

