"""Install only the Web export templates of an official Godot release.

    python3 tools/fetch_godot_web_templates.py 4.7.2 <templates dir>

The release's export_templates.tpz is a 1.3 GB zip of every platform; this
reads its central directory and the web_*.zip entries (~90 MB) with HTTP
range requests and writes them (plus version.txt) into <templates dir>,
e.g. ~/Library/Application Support/Godot/export_templates/4.7.2.stable.
"""

import io
import os
import sys
import urllib.request
import zipfile


class RangeFile(io.RawIOBase):
    """A seekable read-only file over HTTP range requests."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        rng = {"Range": f"bytes={self.pos}-{self.pos + n - 1}"}
        with urllib.request.urlopen(urllib.request.Request(self.url, headers=rng)) as r:
            data = r.read()
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def main():
    version, dest = sys.argv[1], sys.argv[2]
    url = (
        f"https://github.com/godotengine/godot/releases/download/{version}-stable/"
        f"Godot_v{version}-stable_export_templates.tpz"
    )
    z = zipfile.ZipFile(io.BufferedReader(RangeFile(url), buffer_size=1 << 20))
    os.makedirs(dest, exist_ok=True)
    for info in z.infolist():
        name = os.path.basename(info.filename)
        if name == "version.txt" or (name.startswith("web_") and name.endswith(".zip")):
            print(f"  {name} ({info.file_size / 1e6:.1f} MB)", flush=True)
            with z.open(info) as src, open(os.path.join(dest, name), "wb") as out:
                while chunk := src.read(1 << 20):
                    out.write(chunk)


if __name__ == "__main__":
    main()
