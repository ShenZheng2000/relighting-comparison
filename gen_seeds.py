import os

how_many = 100
# how_many = 10

# Set the base file path
base_file = "configs/seeds/exp_4_131_v0.yaml"
output_dir = os.path.dirname(base_file)

# Extract version prefix from the base file
base_name = os.path.basename(base_file)
version_prefix = base_name.rsplit("_", 1)[0] + "_v"

# Read base content
with open(base_file, "r") as f:
    base_content = f.read()

# Generate YAMLs from v0 to v99
for i in range(0, how_many):
    new_file = os.path.join(output_dir, f"{version_prefix}{i}.yaml")

    # Replace or insert seed
    lines = base_content.splitlines()
    updated_lines = []
    seed_replaced = False
    for line in lines:
        if line.strip().startswith("seed:"):
            updated_lines.append(f"seed: {i}")
            seed_replaced = True
        else:
            updated_lines.append(line)
    if not seed_replaced:
        updated_lines.append(f"seed: {i}")

    with open(new_file, "w") as f:
        f.write("\n".join(updated_lines) + "\n")

print(f"✔️ Done generating .yaml with matching seeds.")