"""
Full illustrations for borderless and full art cards, made on this PC.

Scryfall only has these cards' art cut to the old art box. Their full card image has the whole illustration, with the card's
text printed over it: its halftone dots are taken out, LaMa paints out the text (the parts Card Conjurer draws its own frame
and text over), then two Real-ESRGAN models make it 4 times bigger, so it's drawn at the card's size without zooming in.

Runs with the Python of ai_art/.venv (setup.py makes it); the launcher starts it. Card Conjurer asks it on
http://localhost:8081, and the images are kept in local_art/ai_full/<set>/<number>.png (served by the launcher).
"""
import base64
import json
import os
import re
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

import numpy as np
import onnxruntime
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(HERE, "models")
OUTPUT = os.path.join(ROOT, "local_art", "ai_full")
PORT = 8081
ALLOWED_ORIGINS = ("http://localhost:8080", "http://127.0.0.1:8080")
CARD_WIDTH = 2010  # Card Conjurer's card width: the illustration is saved that wide

onnxruntime.set_default_logger_severity(3)
# LaMa on the processor (DirectML can't run its Fourier transforms, a second or two for each part), Real-ESRGAN on the GPU
lama = onnxruntime.InferenceSession(os.path.join(MODELS, "lama.onnx"), providers=["CPUExecutionProvider"])
upscalers = [onnxruntime.InferenceSession(os.path.join(MODELS, name), providers=["DmlExecutionProvider", "CPUExecutionProvider"])
             for name in ("real_esrgan_general_x4v3.onnx", "real_esrgan_x4plus.onnx")]
gpu_lock = threading.Lock()  # one card at a time on the GPU

"""
INPAINTING (LaMa: 512x512 images)
"""


def inpaint(image, mask):
    """image: HxWx3 uint8, mask: HxW bool. Each part of the mask is painted out in a square around it (with as much of the
    picture around it as it covers), at LaMa's 512 pixels."""
    height, width = mask.shape
    result = image.copy()
    for top, bottom, left, right in mask_parts(mask):
        # the square: the part and as much again around it, inside the image
        size = min(max(bottom - top, right - left) * 2, max(height, width))
        center_y, center_x = (top + bottom) // 2, (left + right) // 2
        y0 = int(np.clip(center_y - size // 2, 0, max(0, height - size)))
        x0 = int(np.clip(center_x - size // 2, 0, max(0, width - size)))
        y1, x1 = min(height, y0 + size), min(width, x0 + size)
        crop = Image.fromarray(result[y0:y1, x0:x1])
        crop_mask = Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8))
        image_input = np.asarray(crop.resize((512, 512), Image.LANCZOS), dtype=np.float32).transpose(2, 0, 1)[None] / 255
        mask_input = (np.asarray(crop_mask.resize((512, 512), Image.NEAREST), dtype=np.float32)[None, None] > 0).astype(np.float32)
        output = lama.run(None, {"image": image_input, "mask": mask_input})[0][0].transpose(1, 2, 0)
        filled = Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)).resize((x1 - x0, y1 - y0), Image.LANCZOS)
        # only the masked pixels change (their edge softened over two pixels)
        blend = np.asarray(crop_mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(1)), dtype=np.float32)[..., None] / 255
        region = result[y0:y1, x0:x1].astype(np.float32)
        result[y0:y1, x0:x1] = (region * (1 - blend) + np.asarray(filled, dtype=np.float32) * blend).astype(np.uint8)
    return result


def mask_parts(mask):
    """The bounding boxes of the mask's separate parts (rows of masked runs that touch)."""
    rows = np.where(mask.any(axis=1))[0]
    parts = []
    if len(rows) == 0:
        return parts
    # split by rows first (gaps of empty rows), then each band by columns
    bands = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)
    for band in bands:
        top, bottom = band[0], band[-1] + 1
        columns = np.where(mask[top:bottom].any(axis=0))[0]
        for run in np.split(columns, np.where(np.diff(columns) > 1)[0] + 1):
            parts.append((top, bottom, run[0], run[-1] + 1))
    return parts


"""
HALFTONE (the print's dots, which Real-ESRGAN would make into a grid)
"""


def box_blur(values, size):
    padding = size // 2
    sums = np.pad(np.pad(values, padding, mode="edge").cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return (sums[size:, size:] - sums[:-size, size:] - sums[size:, :-size] + sums[:-size, :-size]) / (size * size)


def descreen(image):
    """The halftone's dots are sharp peaks in the picture's frequencies, past the picture's own detail (0.10 cycles a pixel): each
    one is brought down to the level around it, so the rest of the detail stays (blurring the dots away took it with them)."""
    pixels = image.astype(np.float32)
    height, width = pixels.shape[:2]
    radius = np.sqrt(np.fft.fftfreq(height)[:, None] ** 2 + np.fft.fftfreq(width)[None, :] ** 2)
    for channel in range(3):
        frequencies = np.fft.fft2(pixels[..., channel])
        magnitude = np.abs(frequencies)
        level = np.log1p(magnitude)
        around = np.fft.ifftshift(box_blur(np.fft.fftshift(level), 21))
        peaks = ((level - around) > 0.4) & (radius > 0.10)
        peaks = np.fft.ifftshift(box_blur(np.fft.fftshift(peaks.astype(np.float32)), 5)) > 0
        frequencies[peaks] *= np.expm1(around[peaks]) / np.maximum(magnitude[peaks], 1e-6)
        pixels[..., channel] = np.real(np.fft.ifft2(frequencies))
    return np.clip(pixels, 0, 255).astype(np.uint8)


"""
UPSCALING (Real-ESRGAN: 128x128 tiles)
"""


def upscale(image):
    """image: HxWx3 uint8 -> 4 times bigger: half Real-ESRGAN general x4v3 (it keeps the brushwork, but makes up grain on smooth
    or compressed scans) and half x4plus (clean edges, but it smooths the brushwork away and oversharpens)."""
    return np.mean([upscale_with(image, upscaler) for upscaler in upscalers], axis=0).round().astype(np.uint8)


def upscale_with(image, upscaler):
    """In overlapping tiles blended together."""
    tile, overlap, scale = 128, 16, 4
    height, width = image.shape[:2]
    padded = np.pad(image, ((0, max(0, tile - height)), (0, max(0, tile - width)), (0, 0)), mode="reflect")
    padded_height, padded_width = padded.shape[:2]
    output = np.zeros((padded_height * scale, padded_width * scale, 3), dtype=np.float32)
    weights = np.zeros((padded_height * scale, padded_width * scale, 1), dtype=np.float32)
    # each tile weighs less toward its edges, so the seams between them don't show
    ramp = np.minimum(np.arange(tile * scale) + 1, np.arange(tile * scale)[::-1] + 1).astype(np.float32)
    tile_weight = np.minimum(np.minimum.outer(ramp, ramp), overlap * scale)[..., None]
    step = tile - overlap
    ys = list(range(0, padded_height - tile, step)) + [padded_height - tile]
    xs = list(range(0, padded_width - tile, step)) + [padded_width - tile]
    for y in ys:
        for x in xs:
            tile_input = padded[y:y + tile, x:x + tile].astype(np.float32).transpose(2, 0, 1)[None] / 255
            tile_output = upscaler.run(None, {"image": tile_input})[0][0].transpose(1, 2, 0)
            output[y * scale:(y + tile) * scale, x * scale:(x + tile) * scale] += tile_output * tile_weight
            weights[y * scale:(y + tile) * scale, x * scale:(x + tile) * scale] += tile_weight
    output = output / weights
    return np.clip(output[:height * scale, :width * scale], 0, 1) * 255


"""
THE CARDS
"""


def full_art(image_url, cover=None, rects=()):
    """The card's full illustration: what Card Conjurer's frame and text cover (cover: an image of the card's size, white where
    they do, and/or rects: x, y, width, height as fractions of the card) and its transparent round corners painted out, 4 times
    bigger, then saved at the card's width."""
    with urllib.request.urlopen(urllib.request.Request(image_url, headers={"User-Agent": "CardConjurer/AI full art"}), timeout=30) as response:
        card = Image.open(BytesIO(response.read())).convert("RGBA")
    pixels = np.asarray(card)
    width, height = card.size
    covered = Image.new("L", (width, height), 0)
    if cover is not None:
        covered = cover.convert("L").resize((width, height), Image.BILINEAR)
    # (a little wider: the printed text's soft edges, and the frames not quite where the printed ones are)
    covered = np.asarray(covered.point(lambda value: 255 if value > 40 else 0).filter(ImageFilter.MaxFilter(7))) > 0
    mask = (pixels[..., 3] < 250) | covered
    for x, y, rect_width, rect_height in rects:
        x0, y0 = max(0, int(x * width)), max(0, int(y * height))
        x1, y1 = min(width, int(np.ceil((x + rect_width) * width))), min(height, int(np.ceil((y + rect_height) * height)))
        mask[y0:y1, x0:x1] = True
    # (with a very light blur after: what's left of the dots would come out as grain, and more blur than this smooths the brushwork)
    image = np.asarray(Image.fromarray(descreen(pixels[..., :3])).filter(ImageFilter.GaussianBlur(0.5)))
    image = inpaint(image.copy(), mask)
    big = Image.fromarray(upscale(image))
    return big.resize((CARD_WIDTH, round(CARD_WIDTH * height / width)), Image.LANCZOS)


def output_path(set_code, number, back):
    if not re.fullmatch(r"[a-z0-9]{2,8}", set_code) or not re.fullmatch(r"[A-Za-z0-9★†\-]{1,12}", number):
        raise ValueError("not a card")
    return os.path.join(OUTPUT, set_code, number + ("-back" if back else "") + ".png")


def watch_parent(pid):
    """Stops when the launcher that started it does (closing its window doesn't stop it otherwise)."""
    if os.name != "nt":
        return
    import ctypes
    handle = ctypes.windll.kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not handle:
        os._exit(0)

    def wait():
        ctypes.windll.kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
        os._exit(0)
    threading.Thread(target=wait, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    def cors(self):
        origin = self.headers.get("Origin", "")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def answer(self, status, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.end_headers()

    def do_GET(self):
        if self.path == "/status":
            self.answer(200, {"ready": True, "upscaler": upscalers[0].get_providers()[0]})
        else:
            self.answer(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/full-art":
            return self.answer(404, {"error": "not found"})
        try:
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            image_url = request["image"]
            # only Scryfall's card images
            if not re.match(r"https://cards\.scryfall\.io/", image_url):
                raise ValueError("only Scryfall card images")
            path = output_path(str(request["set"]).lower(), str(request["number"]), bool(request.get("back")))
            rects = [[float(value) for value in rect][:4] for rect in request.get("rects", [])]
            cover = None
            if request.get("cover"):
                cover = Image.open(BytesIO(base64.b64decode(request["cover"].split(",", 1)[1])))
            if not os.path.exists(path) or request.get("redo"):
                with gpu_lock:
                    image = full_art(image_url, cover, rects)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    image.save(path + ".part.png")
                    os.replace(path + ".part.png", path)
            self.answer(200, {"path": "/" + os.path.relpath(path, ROOT).replace(os.sep, "/")})
        except Exception as error:
            self.answer(500, {"error": str(error)})

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # python server.py test <scryfall png url> <output.png> [x,y,w,h ...]
        rects = [[float(value) for value in rect.split(",")] for rect in sys.argv[4:]]
        full_art(sys.argv[2], rects=rects).save(sys.argv[3])
        sys.exit()
    if len(sys.argv) > 2 and sys.argv[1] == "--parent":
        watch_parent(int(sys.argv[2]))
    server = ThreadingHTTPServer(("localhost", PORT), Handler)
    print("AI full art on http://localhost:%s (upscaling on %s)" % (PORT, upscalers[0].get_providers()[0]))
    server.serve_forever()
