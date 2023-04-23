from pathlib import Path

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def _dir_files_cleaned(dir: Path, sort = True):
    files = []
    for i in dir.iterdir():
        if i.is_file() and not i.name.startswith("."):
            files.append(i)

    if sort:
        files.sort()
    return files


class StaticDataset(Dataset):
    def __init__(self, items):
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
            self,
            instance_data_root,
            instance_prompt,
            tokenizer,
    ):
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        self.instance_images_path = _dir_files_cleaned(self.instance_data_root)
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = { }
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        # example["instance_images"] = self.image_transforms(instance_image)
        example["instance_images"] = instance_image
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            truncation = True,
            padding = "max_length",
            max_length = self.tokenizer.model_max_length,
            return_tensors = "pt",
        ).input_ids

        return example


class TransformedDataset(Dataset):
    def __init__(self, src: Dataset, transform):
        self.src = src
        self.transform = transform

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        item = self.src[index]
        item["instance_images"] = self.transform(item["instance_images"])
        return item


def transforms_center_crop(size):
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation = transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def transforms_random_crop(size):
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation = transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = { }
        example["prompt"] = self.prompt
        example["index"] = index
        return example
