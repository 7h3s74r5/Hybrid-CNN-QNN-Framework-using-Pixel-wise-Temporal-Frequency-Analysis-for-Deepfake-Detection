train_files, val_files = train_test_split(

    files,

    test_size=0.2,

    stratify=labels,

    random_state=42

)

print("Training:",len(train_files))

print("Validation:",len(val_files))
