"""
Sets up the AI helper that makes the full illustrations of borderless and full art cards (server.py), once:
- a Python environment of its own in ai_art/.venv, with ONNX Runtime (DirectML: any DirectX 12 graphics card, no CUDA needed)
- the models in ai_art/models, checked against their known contents:
  - LaMa (painting out the text), from OpenCV's models on Hugging Face (Apache 2.0)
  - Real-ESRGAN general x4v3 and x4plus (4 times bigger, half each), as exported by Qualcomm AI Hub (BSD 3-Clause)
  - ISNet (the illustration's subject, for what comes out of the frame), as exported by rembg (Apache 2.0)

Run it with Python 3.10 or newer: python ai_art/setup.py. The launcher starts the helper from then on.
"""
import hashlib
import os
import subprocess
import sys
import urllib.request
import venv
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(HERE, ".venv")
MODELS = os.path.join(HERE, "models")
VENV_PYTHON = os.path.join(VENV, "Scripts", "python.exe") if os.name == "nt" else os.path.join(VENV, "bin", "python")

LAMA = ("https://huggingface.co/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx",
        "7df918ac3921d3daf0aae1d219776cf0dc4e4935f035af81841b40adcf74fdf2")
ISNET = ("https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
         "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a")
UPSCALERS = {
    "real_esrgan_general_x4v3": "fc094329b24a00c936ebc1cc4c51bc59633e7dc36f4a553344342171cbd3c7cf",
    "real_esrgan_x4plus": "05aa31323c6a7e831493171cac970e398697709cae7e7ae8e0760e6da4d64ca4",
}
UPSCALER_URL = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/{0}/releases/v0.64.0/{0}-onnx-float.zip"


def download(url, sha256, path):
    print("Downloading", url)
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, open(path + ".part", "wb") as file:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            file.write(chunk)
    if digest.hexdigest() != sha256:
        os.remove(path + ".part")
        sys.exit("The download isn't the expected file (" + url + "), nothing was installed from it.")
    os.replace(path + ".part", path)


def main():
    if not os.path.exists(VENV_PYTHON):
        print("Making the Python environment in", VENV)
        venv.create(VENV, with_pip=True)
    subprocess.check_call([VENV_PYTHON, "-m", "ensurepip", "--upgrade"], stdout=subprocess.DEVNULL)  # (an environment made without pip)
    subprocess.check_call([VENV_PYTHON, "-m", "pip", "install", "--quiet", "--upgrade", "-r", os.path.join(HERE, "requirements.txt")])

    os.makedirs(MODELS, exist_ok=True)
    if not os.path.exists(os.path.join(MODELS, "lama.onnx")):
        download(*LAMA, os.path.join(MODELS, "lama.onnx"))
    if not os.path.exists(os.path.join(MODELS, "isnet.onnx")):
        download(*ISNET, os.path.join(MODELS, "isnet.onnx"))
    for model, sha256 in UPSCALERS.items():
        if not os.path.exists(os.path.join(MODELS, model + ".onnx")):
            archive = os.path.join(MODELS, model + ".zip")
            download(UPSCALER_URL.format(model), sha256, archive)
            with zipfile.ZipFile(archive) as zip_file:
                for name in (model + ".onnx", model + ".data"):
                    with open(os.path.join(MODELS, name), "wb") as file:
                        file.write(zip_file.read(model + "-onnx-float/" + name))
            os.remove(archive)
    print("Ready. The launcher starts the AI helper with Card Conjurer.")


if __name__ == "__main__":
    main()
