import numpy as np
from sys import maxunicode
from PIL import Image, ImageDraw, ImageFont
from typing import List

class PixelTokenizer():
    def __new__(cls, font_path: str=None, color_mode: str="L", disable_cache: bool=False):
        if disable_cache:
            return ImageTokenizer(font_path=font_path, color_mode=color_mode)
        else:
            return MatrixTokenizer(font_path=font_path, color_mode=color_mode)


class TokenizedPixelSequence():
    def __init__(self, data: np.array, raw_text: str = None, color_mode: str = "L"):
        self.data: np.array = data
        self.text: str = raw_text
        self.color_mode: str = color_mode


    def to_patches(self, patch_length: int=16, pad_color=255) -> np.array:
        if self.color_mode == "L":
            patches = [self.data[:,i:i+patch_length] for i in range(0, self.data.shape[1], patch_length)]
            patches[-1] = np.hstack([patches[-1], np.full((self.data.shape[0], patch_length-patches[-1].shape[1]), pad_color)])
            return np.array(patches)
        else: # modes RGB and RGBA
            patches = []
            for idx_color_channel in range(self.data.shape[2]): # works with any number of channels (RGB, RGBA, ...)
                channel_patches = [self.data[:, :, idx_color_channel][:, i:i + patch_length] for i in range(0, self.data.shape[1], patch_length)]
                channel_patches[-1] = np.hstack([channel_patches[-1], np.full((self.data.shape[0], patch_length-channel_patches[-1].shape[1]), pad_color)])
                patches.append(channel_patches)
            return np.stack(patches, axis=-1) # stack per-channel-patches to full image patches

    @classmethod
    def from_patches(cls, patches: np.array):
        return cls(data=np.hstack(patches))

    def to_image(self) -> Image:
        return Image.fromarray(np.uint8(self.data))

    def to_numpy(self) -> np.array:
        return self.data

    def show(self) -> None:
        self.to_image().show()

    def __repr__(self):
        if self.text:
            return f"<TokenizedPixelSequence with shape {self.data.shape} and raw text '{self.text}'>"
        else:
            return f"<TokenizedPixelSequence with shape {self.data.shape}>"


class ImageTokenizer():
    COLOR_MODES = ["L", "RGB", "RGBA"]

    def __init__(self, font_path: str=None, color_mode: str = "L"):
        self.font: ImageFont = ImageFont.truetype(font_path, encoding="unic") if font_path else ImageFont.load_default()
        if not color_mode in ImageTokenizer.COLOR_MODES:
            raise ValueError(f"color_mode must be one of {ImageTokenizer.COLOR_MODES}. Unexpected value {color_mode}")
        self.color_mode: str = color_mode


    def tokenize(self, raw_text: str) -> TokenizedPixelSequence:
        img_size = (int(self.font.getlength(raw_text)), 15)
        img = Image.new(mode=self.color_mode, size=img_size, color="white")
        draw_obj = ImageDraw.Draw(img)
        draw_obj.text((0, 0), raw_text, fill="black", font=self.font)

        return TokenizedPixelSequence(np.array(img), raw_text, color_mode = self.color_mode)

    def __repr__(self):
        return f"<ImageTokenizer object with font: {' '.join(self.font.getname())} and color_mode: {self.color_mode}>"


class MatrixTokenizer():
    def __init__(self, font_path: str=None, color_mode:str ="L"):
        self.character_tokenizer: ImageTokenizer = ImageTokenizer(font_path=font_path, color_mode=color_mode)
        self.lookup_table: List = [None] * maxunicode

    def _get_from_lookup_table(self, character: str) -> np.array:
        if self.lookup_table[ord(character)] is None:
            self.lookup_table[ord(character)] = self.character_tokenizer.tokenize(character).data
        return self.lookup_table[ord(character)]

    def tokenize(self, raw_text: str) -> TokenizedPixelSequence:
        return TokenizedPixelSequence(np.hstack([self._get_from_lookup_table(ch) for ch in raw_text]), raw_text, color_mode=self.character_tokenizer.color_mode)

    def __repr__(self):
        return f"<MatrixTokenizer object with font {' '.join(self.character_tokenizer.font.getname())} and {sum(x is not None for x in self.lookup_table)} characters in cache>"
