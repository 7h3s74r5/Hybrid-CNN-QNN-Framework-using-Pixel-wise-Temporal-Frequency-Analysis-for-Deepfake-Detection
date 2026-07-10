# ==============================
# NUMBER OF FILES
# ==============================
files = sorted([
    os.path.join(
        SAVE_DIR,
        x
    )
    for x in os.listdir(SAVE_DIR)
    if x.endswith(".npy")
])

print("Total Files:", len(files))

# ==============================
# LABELS EXTRACTION
# ==============================
labels = []

for file in files:

    label = int(
        os.path.basename(file)
        .split("_")[-1]
        .replace(".npy","")
    )

    labels.append(label)

print(Counter(labels))
