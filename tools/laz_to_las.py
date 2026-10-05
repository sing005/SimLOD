"""Decompress .laz point clouds to .las, which SimLOD loads many times faster.

    python tools/laz_to_las.py <file.laz> [more.laz ...] [--out DIR]

Writes <name>.las next to each input (or into --out). Streams in chunks, so
memory use stays small however large the file. Needs:  pip install laspy[laszip]

A .las file is about 6x larger than its .laz (34 bytes per point for the
Rotokare scans), so the script checks free disk space before starting.
"""
import argparse
import os
import shutil
import sys
import time

import laspy

CHUNK = 5_000_000
FAT32_MAX_FILE = 4 * 1024**3 - 1


def is_fat32(path):
    """True if path lies on a FAT32 volume (Windows only; False elsewhere)."""
    if os.name != "nt":
        return False
    import ctypes
    root = os.path.splitdrive(os.path.abspath(path))[0] + "\\"
    fs_name = ctypes.create_unicode_buffer(32)
    ok = ctypes.windll.kernel32.GetVolumeInformationW(root, None, 0, None, None, None, fs_name, 32)
    return bool(ok) and fs_name.value.upper() == "FAT32"


def same_file(a, b):
    try:
        return os.path.samefile(a, b)
    except OSError:             # b does not exist yet
        return False


def convert(src, out_dir):
    dst = os.path.join(out_dir or os.path.dirname(os.path.abspath(src)),
                       os.path.splitext(os.path.basename(src))[0] + ".las")
    # a .las input maps to itself, and opening it for writing emptied it
    if same_file(src, dst):
        print(f"{os.path.basename(src)}: skipped, the output would overwrite the input")
        return False
    with laspy.open(src) as reader:
        header = reader.header
        need = header.offset_to_point_data + header.point_count * header.point_format.size
        free = shutil.disk_usage(os.path.dirname(os.path.abspath(dst))).free
        print(f"{os.path.basename(src)}: {header.point_count / 1e6:.1f} M points, "
              f"needs {need / 1e9:.2f} GB, {free / 1e9:.2f} GB free")
        if need > free * 0.95:
            print("  skipped: not enough disk space")
            return False
        if need > FAT32_MAX_FILE and is_fat32(dst):
            print("  skipped: the destination drive is FAT32, which cannot hold files of 4 GB or more."
                  " Use an NTFS or exFAT drive for this one.")
            return False
        start = time.time()
        # written under another name and renamed when complete, so an
        # interrupted run never leaves a short file that looks finished
        part = dst + ".part"
        try:
            with laspy.open(part, mode="w", header=header, do_compress=False) as writer:
                done = 0
                for chunk in reader.chunk_iterator(CHUNK):
                    writer.write_points(chunk)
                    done += len(chunk)
                    print(f"  {100.0 * done / header.point_count:5.1f}%", end="\r", flush=True)
                if header.evlrs:        # LAS 1.4 extended VLRs; the writer does not copy them
                    writer.write_evlrs(header.evlrs)
                # The input's bounds, not the points' own: laspy's writer resets
                # them and grows them from the points written. SimLOD places its
                # frame at the header's minimum corner, so with a header box
                # looser than the points (the full Rotokare scans) the .las moved
                # the frame, and --view and the haze floor meant something else
                # than on the .laz. recolour_canopy.py keeps them the same way.
                writer.header.mins = header.mins
                writer.header.maxs = header.maxs
        except BaseException:
            if os.path.exists(part):
                os.remove(part)
            raise
        os.replace(part, dst)
        print(f"  wrote {dst} in {time.time() - start:.0f} s")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help=".laz files to convert")
    parser.add_argument("--out", help="output directory (default: next to each input)")
    args = parser.parse_args()
    results = [convert(f, args.out) for f in args.files]   # every file, even after one is skipped
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
