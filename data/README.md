# Dataset Folder

Place UIEB here using paired folders:

```text
data/UIEB/
  train/
    input/
    target/
  val/
    input/
    target/
```

Example:

```text
data/UIEB/train/input/0001.png
data/UIEB/train/target/0001.png
```

The filenames must match because the dataset loader pairs images by filename.
