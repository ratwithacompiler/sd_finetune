import argparse
import contextlib
import itertools
import math
import os
import sys
from functools import partial
from typing import List

import torchvision
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from pathlib import Path

from dataload import LatentZipDataset


def dir_files_cleaned(dir: Path, sort = True):
    files = []
    for i in dir.iterdir():
        if i.is_file() and not i.name.startswith("."):
            files.append(i)

    if sort:
        files.sort()
    return files


from pathlib import Path

import torch
import torchvision.transforms as T
from torchvision.io import read_image, ImageReadMode, encode_png, write_png

IS_DEV = os.environ.get("DEV") == "1"


class ImageDs(Dataset):
    def __init__(self, image_dir, image_read_mode = ImageReadMode.RGB):
        self.image_dir = Path(image_dir)
        if not self.image_dir.exists():
            raise ValueError(f"Instance {self.image_dir} images root doesn't exists.")
        self.image_read_mode = image_read_mode

        self.file_paths = dir_files_cleaned(self.image_dir)
        self._length = len(self.file_paths)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        path = self.file_paths[index]
        return dict(name = path.name, image = read_image(str(path), self.image_read_mode))


import torchvision.transforms.v2


def make_col(args, device, size):
    if not args.augmentation:
        print("no augmentation")
        tfms = transforms.Compose([
            transforms.Resize((size)),
            transforms.RandomCrop((size, size)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.5], [0.5]),
        ])
    else:
        print("default augmentation")
        tfms = transforms.Compose([
            # transforms.RandomPerspective(0.1, 1),
            # transforms.RandomRotation(4, InterpolationMode.NEAREST),
            # torchvision.transforms.v2.RandomIoUCrop(),
            # torchvision.transforms.v2.RandomResizedCrop(size),
            # torchvision.transforms.v2.Resize(128),
            # torchvision.transforms.v2.RandomResizedCrop(128, scale = (0.9,1)),
            torchvision.transforms.v2.RandomPerspective(0.1, p = 0.3),
            torchvision.transforms.v2.RandomAffine(
                # degrees = 0,
                # translate = (0, 0.05),
                degrees = (0, 4),
                translate = (0, 0.05),
                scale = (1.0, 1.075),
                # shear = (0.04, 0.04),
                # scale = (1.0, 1.10),
                # shear = (8, 8),
                # shear = (0.1, 0.1),
                # scale = (1.0, 1.06),
                interpolation = InterpolationMode.BILINEAR,
            ),
            # torchvision.transforms.v2.RandomCrop(128),
            # transforms.Resize((size, size)),
            transforms.Resize((size)),
            transforms.RandomCrop((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness = 0.15),
            transforms.ColorJitter(contrast = 0.15),
            transforms.ColorJitter(saturation = 0.15),
            transforms.ColorJitter(hue = 0.01),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.5], [0.5]),
            # transforms.RandomRotation
            # transforms.ColorJitter
            # transforms.RandomHorizontalFlip
            # transforms.Pad
        ])

    def col(batch):
        images = []
        names = []
        for i in batch:
            img = i["image"]
            img = img.to(device)
            images.append(tfms(img))
            names.append(i["name"])
        return names, torch.stack(images)

    return col


def show_images(imgs, nrow = 4):
    import matplotlib.pyplot as plt

    # fix, axs = plt.subplots(ncols = len(imgs), squeeze = False)
    # for i, img in enumerate(imgs):
    #     img = T.ToPILImage()(img.to('cpu'))
    #     axs[0, i].imshow(np.asarray(img))
    #     axs[0, i].set(xticklabels = [], yticklabels = [], xticks = [], yticks = [])

    grid_img = torchvision.utils.make_grid(imgs, nrow = nrow)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()


def show_image(img_tens: torch.Tensor):
    import matplotlib.pyplot as plt
    plt.imshow(img_tens.permute(1, 2, 0))
    plt.show()


def eprint(*args, **kwargs):
    print(*args, **kwargs, file = sys.stderr)


def cycle_n(n, multi_iterable):
    for _ in range(n):
        for i in multi_iterable:
            yield i


def main_convert(args):
    torch.set_grad_enabled(False)
    torch.manual_seed(123)
    if args.device:
        device = torch.device(args.device)
    else:
        device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    collate_device = device

    if IS_DEV:
        args.input_path = "../dev/data2_smol/"
        args.input_path = "../dev/data1"
        args.input_path = "../dev/data2"
        args.batch_size = 4
        # size = (512, 512)
        args.size = 64
        args.size = 512
        args.make_dirs = True
        # args.latents = True
        args.image = True
        args.number = 4
        collate_device = "cpu"
        pass

    eprint("device:", device)
    eprint("collate_device:", collate_device)

    output_path = Path(args.output_path)
    if not output_path.is_dir():
        if args.make_dirs:
            eprint(f"creating output dir: {str(output_path)!r}")
            output_path.mkdir(parents = True)
        else:
            eprint(f"Error: output_path not found: {str(output_path)!r}")
            exit(1)

    save_image = args.image
    save_latents = args.latents
    if not save_image and not save_latents:
        eprint(f"Error: provide one of --image or --latents!")
        exit(1)

    size = args.size
    vae = AutoencoderKL.from_pretrained(args.vae_name_or_path, subfolder = "vae")
    vae = vae.to(device)

    train_dataset = ImageDs(args.input_path)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        collate_fn = make_col(args, collate_device, size),
        # num_workers = args.num_workers,
    )
    if args.number > 1:
        train_dataloader = cycle_n(args.number, train_dataloader)

    try:
        import tqdm
        progress = partial(tqdm.tqdm, total = math.ceil(len(train_dataset) * args.number / args.batch_size))
    except:
        progress = lambda i: i

    # train_dataset = DreamBoothDataset(args.input_path)
    # def dictcol(batch_dict):
    #     return torch.stack([i["instance_images"] for i in batch_dict])
    # train_dataloader = DataLoader(
    #     TransformedDataset(train_dataset, transforms_random_crop(size)),
    #     batch_size = 1,
    #     shuffle = True,
    #     collate_fn = dictcol,
    # )
    to_int = T.ConvertImageDtype(torch.uint8)

    for pos, (names, batch) in progress(enumerate(train_dataloader)):
        print("pos", pos)
        batch: torch.Tensor
        bs = batch.shape[0]

        if save_latents:
            latents = vae.encode(batch).latent_dist.sample()
        else:
            latents = [None] * bs

        assert len(names) == bs
        assert len(latents) == bs
        assert len(batch) == bs
        for name, lt, pixels in zip(names, latents, batch):
            if lt is not None:
                with atomic(find_filepath(output_path, name, ".pt")) as fp:
                    torch.save(lt, fp)
            if save_image:
                write_pixels = (pixels / 2 + 0.5).clamp(0, 1)
                with atomic(find_filepath(output_path, name, ".png")) as fp:
                    write_png(to_int(write_pixels.to("cpu")), str(fp), 0)


@contextlib.contextmanager
def atomic(filepath, verbose = False):
    # return path.tmp in with and rename to path if finished without exception
    if isinstance(filepath, str):
        tmp = filepath + "tmp"
    else:
        tmp = filepath.with_name(filepath.name + ".tmp")

    yield tmp

    if verbose:
        print("moving {!r} -> {!r}".format(str(tmp), str(filepath)))
    os.rename(tmp, filepath)


def find_filepath(output_path: Path, name_start: str, ext: str, max_tries = 1024) -> Path:
    option = None
    for i in range(1, max_tries + 1):
        option = output_path / f"{name_start}.{i}{ext}"
        if not option.exists():
            return option
    raise ValueError("reached max tries, couldnt find filename", output_path, name_start, ext, option, max_tries)


def show_collate_fn(examples: List[dict]) -> dict:
    input_ids = [example["instance_prompt_ids"] for example in examples if "instance_prompt_ids" in example] or None
    latents = [example["latents"] for example in examples if "latents" in example] or None
    if latents:
        latents = torch.cat(latents)
        latents = latents.to(memory_format = torch.contiguous_format).float()

    if input_ids:
        input_ids = torch.cat(input_ids, dim = 0)

    batch = {
        "input_ids": input_ids,
        "latents":   latents,
    }
    return batch


def main_show(args):
    torch.set_grad_enabled(False)
    if args.device:
        device = torch.device(args.device)
    else:
        device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = LatentZipDataset(args.zip_path)
    train_dataloader = DataLoader(
        ds,
        batch_size = args.batch_size,
        shuffle = True,
        collate_fn = show_collate_fn,
    )

    vae = AutoencoderKL.from_pretrained(args.vae_name_or_path, subfolder = "vae")
    vae = vae.to(device)

    for batch in train_dataloader:
        latents = batch["latents"]
        image = vae.decode(latents.to(device))
        image = image.sample

        # image = (image / 2 + 0.5).clamp(0, 1)
        # image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        # images = (image * 255).round().astype("uint8")
        # import PIL.Image
        # pil_images = [PIL.Image.fromarray(image) for image in images]
        # for pos, i in enumerate(pil_images):
        #     i.save(f"../dev/test/{pos}.png")

        # show_images(batch)
        show_images((image / 2 + 0.5).clamp(0, 1).cpu())
    pass


def setup_args(argv = None):
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    subp = parser.add_subparsers()
    subp.required = True
    convert_parser = subp.add_parser("convert")
    convert_parser.set_defaults(fn = main_convert)
    convert_parser.add_argument("input_path")
    convert_parser.add_argument("output_path")
    convert_parser.add_argument("-a", "--augmentation", action = "store_true")
    convert_parser.add_argument("-n", "--number", type = int, default = 1)
    convert_parser.add_argument("-m", "--make_dirs", action = "store_true")
    convert_parser.add_argument("-d", "--device")
    convert_parser.add_argument("-N", "--num_workers", type = int, default = None)
    convert_parser.add_argument("-P", "--vae_name_or_path", default = "runwayml/stable-diffusion-v1-5")
    convert_parser.add_argument("-b", "--batch_size", type = int, default = 1)
    convert_parser.add_argument("-s", "--size", type = int, default = 512)

    convert_parser.add_argument("-i", "--image", action = "store_true")
    convert_parser.add_argument("-l", "--latents", action = "store_true")

    show_parser = subp.add_parser("show")
    show_parser.set_defaults(fn = main_show)
    show_parser.add_argument("zip_path")
    show_parser.add_argument("-P", "--vae_name_or_path", default = "runwayml/stable-diffusion-v1-5")
    show_parser.add_argument("-b", "--batch_size", type = int, default = 1)
    show_parser.add_argument("-d", "--device")

    return parser.parse_args(argv) if argv else parser.parse_args()


if __name__ == '__main__':
    def _():
        args = setup_args()
        args.fn(args)


    _()
