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
from collections import deque
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
DEBUG = os.environ.get("CC_AI_DEBUG")  # a folder to keep each step's pictures in
if DEBUG:
    os.makedirs(DEBUG, exist_ok=True)

onnxruntime.set_default_logger_severity(3)
# LaMa on the processor (DirectML can't run its Fourier transforms, a second or two for each part), Real-ESRGAN on the GPU
lama = onnxruntime.InferenceSession(os.path.join(MODELS, "lama.onnx"), providers=["CPUExecutionProvider"])
# ISNet (finding the illustration's subject, for what comes out of the frame), when it's there
subject_finder = None
if os.path.exists(os.path.join(MODELS, "isnet.onnx")):
    subject_finder = onnxruntime.InferenceSession(os.path.join(MODELS, "isnet.onnx"), providers=["DmlExecutionProvider", "CPUExecutionProvider"])
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


def upscale(image, general_share):
    """image: HxWx3 uint8 -> 4 times bigger: Real-ESRGAN general x4v3 (it keeps the brushwork, but makes up grain on smooth or
    compressed scans) and x4plus (clean edges, but it smooths the brushwork away and oversharpens), general_share of the first."""
    output = 0
    for upscaler, share in zip(upscalers, (general_share, 1 - general_share)):
        if share > 0:
            output = output + upscale_with(image, upscaler) * share
    return np.clip(output, 0, 255).round().astype(np.uint8)


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


# How much detail is kept: the blur after the halftone's taken out (what's left of the dots would come out as grain, more blur
# smooths the brushwork) and how much of the upscaling is the general model's
DETAIL = {
    "soft": {"blur": 1.1, "general": 1},  # smooth, for scans with little detail (x4plus makes up hair-like strokes when blurred)
    "balanced": {"blur": 0.5, "general": 0.5},
    "detailed": {"blur": 0.3, "general": 1},  # all the brushwork (and grain made up on smooth or compressed scans)
}


def printed_bands(brightness, groups, colors, kept):
    """A planeswalker's ability boxes, light and dark in turn. The printed ones are as tall as the printed text (in the card
    image's language, which may not be the imported card's), so each line between two boxes is moved to where the card image
    changes most, near the frame's. And the printed dark boxes aren't as see-through as the frame's: their opacity is the one
    that leaves the illustration as bright just under each line as just over it (under the light box, with the frame's
    opacity). The color and opacity of each are put in colors and kept."""
    height = brightness.shape[0]
    light, dark = sorted(groups, key=lambda group: -colors[group].mean())
    light_color, dark_color = np.median(colors[light], axis=0), np.median(colors[dark], axis=0)
    light_opacity, dark_opacity = np.median(kept[light]), np.median(kept[dark])
    region = light | dark
    rows = np.where(region.any(axis=1))[0]
    top, bottom = rows[0], rows[-1] + 1
    columns = region[top:bottom].mean(axis=0) > 0.5
    # each row's brightness behind its text (the text's dark, so the brighter part)
    profile = np.percentile(brightness[top:bottom][:, columns], 70, axis=1)
    frame_light = light[top:bottom].sum(axis=1) > dark[top:bottom].sum(axis=1)
    lines = [row for row in range(1, len(frame_light)) if frame_light[row] != frame_light[row - 1]]
    near, side = int(0.03 * height), 8
    moved = []
    for line in lines:
        steps = {row: abs(np.median(profile[row + 2:row + side]) - np.median(profile[row - side:row - 2]))
                 for row in range(max(side, line - near), min(len(profile) - side, line + near))}
        moved.append(max(steps, key=steps.get) if steps else line)
    is_light = np.zeros(len(profile), dtype=bool)
    for start, end, box_light in zip([0] + moved, moved + [len(profile)], [frame_light[0]] + [frame_light[line] for line in lines]):
        is_light[start:end] = box_light
    # the printed dark box as the light one darkened: how much, from small pieces along each line (where the illustration's
    # about the same on both sides of it), the brighter part of each side (not its text)
    darkenings = []
    piece = 12
    column_numbers = np.where(columns)[0]
    for line in moved:
        for start in range(0, len(column_numbers) - piece, piece):
            span = column_numbers[start:start + piece]
            over = np.percentile(brightness[top + max(0, line - side):top + line - 2][:, span], 70)
            under = np.percentile(brightness[top + line + 2:top + line + side][:, span], 70)
            light_side, dark_side = (over, under) if is_light[line - 1] else (under, over)
            if light_side > 40:
                darkenings.append(dark_side / light_side)
    darkening = float(np.clip(np.median(darkenings), 0.5, 1)) if darkenings else None
    region = np.zeros_like(region)
    region[top:bottom] = columns
    light_rows = np.zeros(height, dtype=bool)
    light_rows[top:bottom] = is_light
    light, dark = region & light_rows[:, None], region & ~light_rows[:, None]
    colors[light], colors[dark] = light_color, dark_color
    kept[light], kept[dark] = light_opacity, dark_opacity
    return dark, darkening


"""
COMING OUT OF THE FRAME (what the printed card has over its frame: a character's hair over the title bar...)
"""


def parts(mask):
    """The separate parts of a mask (pixels touching up, down or sideways), each as its rows and columns."""
    height, width = mask.shape
    seen = np.zeros_like(mask)
    found = []
    for start_y, start_x in zip(*np.nonzero(mask)):
        if seen[start_y, start_x]:
            continue
        seen[start_y, start_x] = True
        queue, points = deque([(start_y, start_x)]), []
        while queue:
            y, x = queue.popleft()
            points.append((y, x))
            for next_y, next_x in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= next_y < height and 0 <= next_x < width and mask[next_y, next_x] and not seen[next_y, next_x]:
                    seen[next_y, next_x] = True
                    queue.append((next_y, next_x))
        points = np.array(points)
        found.append((points[:, 0], points[:, 1]))
    return found


def grow(mask, size):
    return np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(size))) > 0


def shrink_and_grow(mask, size):
    """(what's thinner than size goes: the frames' edges not quite where the printed ones are, the printed text's strokes)"""
    return np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(size)).filter(ImageFilter.MaxFilter(size))) > 0


def popped_out(card, layers, opacity):
    """Where the card image isn't the frame (Card Conjurer's, opaque there), in pieces that touch the art window, and that ISNet
    finds are the illustration's subject (not the printed text, symbols, or the frame's edges)."""
    if subject_finder is None:
        return None
    height, width = opacity.shape
    picture = np.asarray(Image.fromarray(descreen(card)).filter(ImageFilter.GaussianBlur(1))).astype(np.float32)
    exact_difference = np.sqrt(((picture - layers[..., :3]) ** 2).sum(axis=2))
    # (against the frame moved up to 3 pixels each way: its lines aren't quite where the printed ones are)
    difference = np.full(opacity.shape, np.inf, dtype=np.float32)
    for shift_y in range(-3, 4):
        for shift_x in range(-3, 4):
            moved = np.roll(layers[..., :3], (shift_y, shift_x), axis=(0, 1))
            difference = np.minimum(difference, np.sqrt(((picture - moved) ** 2).sum(axis=2)))
    opaque = opacity > 0.85
    unlike = (difference > 70) & opaque
    see_through = parts(opacity < 0.08)
    if not see_through:
        return None
    art_window = np.zeros_like(opaque)
    art_window[max(see_through, key=lambda part: len(part[0]))] = True
    # (13 pixels past the window's edge: where the frame's lines around it are, which can look like what crosses them)
    near_window = grow(art_window, 27)
    pieces = np.zeros_like(opaque)
    for rows, columns in parts(shrink_and_grow(unlike, 7)):
        if len(rows) >= 120 and near_window[rows, columns].any():
            pieces[rows, columns] = True
    if not pieces.any():
        return None
    # the subject, around the art window (and the frame just around it)
    rows, columns = np.nonzero(art_window)
    top, bottom = max(0, rows.min() - int(0.08 * height)), min(height, rows.max() + int(0.08 * height))
    left, right = max(0, columns.min() - int(0.04 * width)), min(width, columns.max() + int(0.04 * width))
    around = Image.fromarray(card[top:bottom, left:right]).resize((1024, 1024), Image.LANCZOS)
    found = subject_finder.run(None, {"input_image": ((np.asarray(around, dtype=np.float32) / 255 - 0.5).transpose(2, 0, 1)[None])})[0][0, 0]
    found = (found - found.min()) / (found.max() - found.min() + 1e-6)
    subject = np.zeros_like(opaque)
    subject[top:bottom, left:right] = np.asarray(Image.fromarray((found * 255).astype(np.uint8)).resize((right - left, bottom - top), Image.BILINEAR)) > 180
    # each piece grown along the subject where it isn't the frame (as it is: over the frame's lines it crosses), and with the thin
    # bits around it (hair's ends)
    along = subject & (exact_difference > 50) & opaque
    pieces &= subject
    for _ in range(20):
        grown = pieces | (grow(pieces, 3) & along)
        if grown.sum() == pieces.sum():
            break
        pieces = grown
    return pieces


def full_art(image_url, frame=None, rects=(), detail="balanced", pop_out=False):
    """The card's full illustration, from its card image, 4 times bigger and saved at the card's width. frame: Card Conjurer's
    frame (an RGBA image of the card), drawn over it the way the printed one was:
    - where it's opaque, the picture's painted out (it doesn't show)
    - where it's see-through (planeswalkers' ability boxes, borderless cards' text boxes), the printed box is taken off the
      picture (the frame's color at its opacity), so the illustration under it shows again through the frame's own box, and only
      the printed text's painted out
    - the rest is the illustration as printed
    rects: the text boxes (x, y, width, height as fractions of the card), where printed text is looked for. Without a frame they're
    painted out whole. The card image's transparent round corners are painted out too.
    pop_out: what comes out of the frame on the printed card is kept, and returned on its own too (else None)."""
    settings = DETAIL[detail]
    popped = None
    with urllib.request.urlopen(urllib.request.Request(image_url, headers={"User-Agent": "CardConjurer/AI full art"}), timeout=30) as response:
        card = Image.open(BytesIO(response.read())).convert("RGBA")
    pixels = np.asarray(card)
    width, height = card.size
    image = np.asarray(Image.fromarray(descreen(pixels[..., :3])).filter(ImageFilter.GaussianBlur(settings["blur"]))).astype(np.float32)
    mask = pixels[..., 3] < 250
    text_boxes = np.zeros((height, width), dtype=bool)
    for x, y, rect_width, rect_height in rects:
        x0, y0 = max(0, int(x * width)), max(0, int(y * height))
        x1, y1 = min(width, int(np.ceil((x + rect_width) * width))), min(height, int(np.ceil((y + rect_height) * height)))
        text_boxes[y0:y1, x0:x1] = True
    if frame is None:
        mask |= text_boxes
    else:
        layers = np.asarray(frame.convert("RGBA").resize((width, height), Image.BILINEAR)).astype(np.float32)
        opacity = layers[..., 3] / 255
        # (the opaque parts a little wider: the frames aren't quite where the printed ones are)
        opaque = np.asarray(Image.fromarray(((opacity > 0.85) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(7))) > 0
        see_through = (opacity > 0.08) & ~opaque
        kept = np.clip(opacity, 0, 0.85)
        # each of the frame's see-through boxes (one color and opacity) is taken off where the printed one is
        brightness = image.mean(axis=2)
        boxes = np.round(layers / 16).astype(np.int64)
        box_keys = np.where(see_through, ((boxes[..., 0] * 17 + boxes[..., 1]) * 17 + boxes[..., 2]) * 17 + boxes[..., 3], -1)
        keys, counts = np.unique(box_keys[see_through], return_counts=True)
        groups = [box_keys == key for key, count in zip(keys, counts) if count >= 1500]
        colors = layers[..., :3].copy()
        if len(groups) == 2:
            dark, darkening = printed_bands(brightness, groups, colors, kept)
            if darkening:
                # (lightened back to the light box, then taken off like it)
                light = (groups[0] | groups[1]) & ~dark
                image = np.where(dark[..., None], image / darkening, image)
                colors[dark], kept[dark] = np.median(colors[light & see_through], axis=0), np.median(kept[light & see_through])
        untinted = (image - kept[..., None] * colors) / (1 - kept[..., None])
        image = np.where(see_through[..., None], np.clip(untinted, 0, 255), image)
        mask |= opaque
        # what comes out of the frame (kept as printed, not painted out)
        if pop_out:
            popped = popped_out(pixels[..., :3], layers, opacity)
            if popped is not None:
                mask &= ~popped
        if DEBUG:
            frame.save(os.path.join(DEBUG, "frame.png"))
            Image.fromarray(image.astype(np.uint8)).save(os.path.join(DEBUG, "untinted.png"))
        # the printed text in the text boxes, where the frame doesn't hide it: what stands out from what's around it
        brightness = image.mean(axis=2)
        around = box_blur(brightness, 9)
        printed = (np.abs(brightness - around) > 28) & text_boxes & ~opaque
        mask |= np.asarray(Image.fromarray((printed * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))) > 0
    image = inpaint(image.astype(np.uint8), mask)
    if DEBUG:
        Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(DEBUG, "mask.png"))
        Image.fromarray(image).save(os.path.join(DEBUG, "painted.png"))
    big = Image.fromarray(upscale(image, settings["general"])).resize((CARD_WIDTH, round(CARD_WIDTH * height / width)), Image.LANCZOS)
    if popped is None or not popped.any():
        return big, None
    # the part that comes out of the frame, on its own (its edge softened over a pixel), to be drawn over the frame
    edge = Image.fromarray((popped * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7)).resize(big.size, Image.BILINEAR)
    layer = big.convert("RGBA")
    layer.putalpha(edge)
    return big, layer


def output_path(set_code, number, back, detail):
    """local_art/ai_full/<set>/<number>[-back][-soft|-detailed].png (balanced, the usual one, without a suffix)"""
    if not re.fullmatch(r"[a-z0-9]{2,8}", set_code) or not re.fullmatch(r"[A-Za-z0-9★†\-]{1,12}", number) or detail not in DETAIL:
        raise ValueError("not a card")
    suffix = ("-back" if back else "") + ("" if detail == "balanced" else "-" + detail)
    return os.path.join(OUTPUT, set_code, number + suffix + ".png")


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
            detail = str(request.get("detail", "balanced"))
            path = output_path(str(request["set"]).lower(), str(request["number"]), bool(request.get("back")), detail)
            rects = [[float(value) for value in rect][:4] for rect in request.get("rects", [])]
            frame = None
            if request.get("frame"):
                frame = Image.open(BytesIO(base64.b64decode(request["frame"].split(",", 1)[1])))
            # what comes out of the frame: <the illustration's name>-popout.png, when the printed card has any
            popout_path = path[:-len(".png")] + "-popout.png"
            if not os.path.exists(path) or request.get("redo"):
                with gpu_lock:
                    image, popout = full_art(image_url, frame, rects, detail, pop_out=frame is not None)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    image.save(path + ".part.png")
                    os.replace(path + ".part.png", path)
                    if popout is not None:
                        popout.save(popout_path + ".part.png")
                        os.replace(popout_path + ".part.png", popout_path)
                    elif os.path.exists(popout_path):
                        os.remove(popout_path)
            web = lambda file: "/" + os.path.relpath(file, ROOT).replace(os.sep, "/")
            self.answer(200, {"path": web(path), "popout": web(popout_path) if os.path.exists(popout_path) else None})
        except Exception as error:
            self.answer(500, {"error": str(error)})

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # python server.py test <scryfall png url> <output.png> <soft|balanced|detailed> [x,y,w,h ...]
        rects = [[float(value) for value in rect.split(",")] for rect in sys.argv[5:]]
        full_art(sys.argv[2], rects=rects, detail=sys.argv[4])[0].save(sys.argv[3])
        sys.exit()
    if len(sys.argv) > 2 and sys.argv[1] == "--parent":
        watch_parent(int(sys.argv[2]))
    server = ThreadingHTTPServer(("localhost", PORT), Handler)
    print("AI full art on http://localhost:%s (upscaling on %s)" % (PORT, upscalers[0].get_providers()[0]))
    server.serve_forever()
