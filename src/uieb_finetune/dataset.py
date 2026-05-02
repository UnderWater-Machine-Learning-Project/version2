from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PairedUIEBDataset(Dataset):
    """Loads underwater input images and their clean/reference targets."""

    def __init__(self, root, split, image_size):
        self.root = Path(root)
        self.split = split
        self.input_dir = self.root / split / "input"
        self.target_dir = self.root / split / "target"

        self.input_paths = self._find_images(self.input_dir)
        if not self.input_paths:
            raise FileNotFoundError(f"No images found in {self.input_dir}")

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index):
        input_path = self.input_paths[index]
        target_path = self.target_dir / input_path.name

        if not target_path.exists():
            raise FileNotFoundError(
                f"Missing target image for {input_path.name}: {target_path}"
            )

        underwater = self._load_rgb(input_path)
        target = self._load_rgb(target_path)

        return {
            "underwater": self.transform(underwater),
            "target": self.transform(target),
            "name": input_path.name,
        }

    @staticmethod
    def _find_images(folder):
        if not folder.exists():
            raise FileNotFoundError(f"Dataset folder does not exist: {folder}")

        return sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    @staticmethod
    def _load_rgb(path):
        with Image.open(path) as image:
            return image.convert("RGB")
