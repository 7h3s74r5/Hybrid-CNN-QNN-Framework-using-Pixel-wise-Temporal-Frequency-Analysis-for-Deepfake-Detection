
labels = []

for file in files:

    label = int(
        os.path.basename(file)
        .split("_")[-1]
        .replace(".npy","")
    )

    labels.append(label)

print(Counter(labels))
