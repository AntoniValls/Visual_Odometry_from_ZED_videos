import json

# Change this to your actual input file path
input_path = '../datasets/BIEL/00/odometryAdria.txt'
output_path = '../datasets/BIEL/00/OdA.txt'

# Output list
poses = []

with open(input_path, 'r') as file:
    for line in file:
        # Split and convert to float
        values = list(map(float, line.strip().split()))
        if len(values) != 7:
            continue  # skip malformed lines

        tx, ty, tz = values[0:3]
        qx, qy, qz, qw = values[3:7]

        pose = {
            "pose": {
                "translation": [tx, ty, tz],
                "quaternion": [qx, qy, qz, qw]
            }
        }

        poses.append(pose)

# Write each pose as a line to the .txt file
with open(output_path, 'w') as out_file:
    for pose in poses:
        out_file.write(json.dumps(pose) + '\n')

print(f"Successfully written {len(poses)} formatted poses to '{output_path}'.")
