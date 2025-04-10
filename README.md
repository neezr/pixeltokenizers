# pixeltokenizers

A library to tokenize text as patches of images that render text very fast due to caching on character level.\
Use these tokenizers to train Pixel Language Models as described in [Language Modelling with Pixels (Rust et. al 2020)](https://arxiv.org/abs/2207.06991).

## Example Usage:

``` python
from pixeltokenizers import PixelTokenizer

tokenizer = PixelTokenizer(r"C:\Windows\Fonts\arial.ttf")
>>> <MatrixTokenizer object with font Arial Regular and 0 characters in cache>


tokenizer.tokenize("Example Text").to_numpy().shape
>>> (15, 63)

tokenizer.tokenize("Example Text").to_image().show()
>>> <PIL.Image.Image image mode=L size=63x15 at 0x25624B9CE50>

patches = tokenizer.tokenize("Example Text").to_patches(patch_length = 32)
patches[0].shape
>>> (15, 32)

```

## Known Issues
- Caching does not work on languages that connect letters (e.g. Arabic)
  - use ```PixelTokenizer(disable_cache=True)```
- Bidirectional and right-to-left text (e.g. Hebrew) is not handled well 
  - use external libraries for handling bidirectional text, such as ```python-bidi```
- Default fonts do not display emoji
