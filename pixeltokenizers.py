import numpy as np
from sys import maxunicode
from PIL import Image, ImageDraw, ImageFont
from typing import List

class PixelTokenizer():
    def __new__(cls, font_path=None, disable_cache=False):
        if disable_cache:
            return ImageTokenizer(font_path=font_path)
        else:
            return MatrixTokenizer(font_path=font_path)


class TokenizedPixelSequence():
    def __init__(self, data: np.array, raw_text: str = None):
        self.data: np.array = data
        self.text: str = raw_text

    def to_patches(self, patch_length=16, pad_color=255) -> np.array:
        patches = [self.data[:,i:i+patch_length] for i in range(0, self.data.shape[1], patch_length)]
        patches[-1] = np.hstack([patches[-1], np.full((self.data.shape[0], patch_length-patches[-1].shape[1]), pad_color)])
        return np.array(patches)

    @classmethod
    def from_patches(cls, patches: np.array):
        return cls(data=np.hstack(patches))

    def to_image(self) -> Image:
        return Image.fromarray(np.uint8(self.data))

    def to_numpy(self) -> np.array:
        return self.data

    def __repr__(self):
        if self.text:
            return f"<TokenizedPixelSequence with shape {self.data.shape} and raw text '{self.text}'>"
        else:
            return f"<TokenizedPixelSequence with shape {self.data.shape}>"


class ImageTokenizer():
    def __init__(self, font_path: str=None):
        self.font: ImageFont = ImageFont.truetype(font_path, encoding="unic") if font_path else ImageFont.load_default()

    def tokenize(self, raw_text: str) -> TokenizedPixelSequence:
        img_size = (int(self.font.getlength(raw_text)), 15)
        img = Image.new("L", size=img_size, color="white")
        draw_obj = ImageDraw.Draw(img)
        draw_obj.text((0, 0), raw_text, fill="black", font=self.font)

        return TokenizedPixelSequence(np.array(img), raw_text)

    def __repr__(self):
        return f"<ImageTokenizer object with font: {' '.join(self.font.getname())}>"


class MatrixTokenizer():
    def __init__(self, font_path: str=None):
        self.character_tokenizer: ImageTokenizer = ImageTokenizer(font_path=font_path)
        self.lookup_table: List = [None] * maxunicode

    def _get_from_lookup_table(self, character: str) -> np.array:
        if self.lookup_table[ord(character)] is None:
            self.lookup_table[ord(character)] = self.character_tokenizer.tokenize(character).data
        return self.lookup_table[ord(character)]

    def tokenize(self, raw_text: str) -> TokenizedPixelSequence:
        return TokenizedPixelSequence(np.hstack([self._get_from_lookup_table(ch) for ch in raw_text]), raw_text)

    def __repr__(self):
        return f"<MatrixTokenizer object with font {' '.join(self.character_tokenizer.font.getname())} and {sum(x is not None for x in self.lookup_table)} characters in cache>"

