import io
import zipfile
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image


def dir_files_cleaned(dir: Path, sort = True):
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
            instance_prompt = None,
            tokenizer = None,
    ):
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        self.instance_images_path = dir_files_cleaned(self.instance_data_root)
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt

        from torchvision.io import ImageReadMode
        self.image_read_mode = ImageReadMode.RGB

    def __len__(self):
        return self.num_instance_images

    def __getitem__(self, index):
        example = { }

        instance_image = read_image(str(self.instance_images_path[index]), self.image_read_mode)
        example["instance_images"] = instance_image

        if self.instance_prompt is not None:
            example["instance_prompt_ids"] = self.tokenizer(
                self.instance_prompt,
                truncation = True,
                padding = "max_length",
                max_length = self.tokenizer.model_max_length,
                return_tensors = "pt",
            ).input_ids

        return example


class RepeatedDataset(Dataset):
    def __init__(self, ds: Dataset, repeat: int):
        self.ds = ds
        self.ds_len = len(ds)
        self.repeat = int(repeat)
        self._length = self.ds_len * self.repeat

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return self.ds[index % self.ds_len]


class LatentZipDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(self, zip_path,
                 instance_prompt = None,
                 tokenizer = None,
                 ):
        self.tokenizer = tokenizer

        self.zip_path = Path(zip_path)
        if not self.zip_path.exists():
            raise ValueError(f"Latents {self.zip_path} zip_path doesn't exists.")

        self.zip_file = zipfile.ZipFile(zip_path, "r")
        filenames = self.zip_file.namelist()
        self.filenames = [i for i in filenames if not i.startswith(".")]
        # self.filenames = [i for i in filenames if not i.startswith("mpv-shot0004_b")]

        self.num_instance_images = len(self.filenames)
        self.instance_prompt = instance_prompt

    def __len__(self):
        return self.num_instance_images

    def __getitem__(self, index):
        example = { }

        torch_data = self.zip_file.read(self.filenames[index])
        data = torch.load(io.BytesIO(torch_data), map_location = "cpu")
        assert data.shape[0] == 4
        assert len(data.shape) == 3
        data = data[None]
        example["latents"] = data

        if self.instance_prompt is not None:
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
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(size, interpolation = transforms.InterpolationMode.BILINEAR, antialias = True),
            transforms.CenterCrop(size),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def transforms_random_crop(size):
    return transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(size, interpolation = transforms.InterpolationMode.BILINEAR, antialias = True),
            transforms.RandomCrop(size),
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
