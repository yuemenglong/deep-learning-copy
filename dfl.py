from interact import interact as io
import os


def dfl_train(src_aligned, dst_aligned, model_dir, model="SAE"):
    cmd = "train"
    args = {
        "--training-data-src-dir": src_aligned,
        "--training-data-dst-dir": dst_aligned,
        "--model-dir": model_dir,
        "--model": model,
    }
    dfl_exec(cmd, args)


def dfl_convert(input_dir, output_dir, aligned_dir, model_dir, enable_predef=True, model="SAE"):
    cmd = "convert"
    args = {
        "--input-dir": input_dir,
        "--output-dir": output_dir,
        "--aligned-dir": aligned_dir,
        "--model-dir": model_dir,
        "--model": model,
    }
    if enable_predef:
        args["--enable-predef"] = ""
    dfl_exec(cmd, args)


def dfl_sort_by_hist(input_dir):
    cmd = "sort"
    args = {
        "--input-dir": input_dir,
        "--by": "hist",
    }
    dfl_exec(cmd, args)


def dfl_video_from_sequence(input_dir, output_file, reference_file):
    cmd = "videoed video-from-sequence"
    args = {
        "--input-dir": input_dir,
        "--output-file": output_file,
        "--reference-file": reference_file,
    }
    dfl_exec(cmd, args)


def dfl_extract_video(input_file, output_dir, fps=0):
    # "%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py"
    # videoed
    # extract - video ^
    # --input - file
    # "%WORKSPACE%\data_dst.*" ^
    # --output - dir
    # "%WORKSPACE%\data_dst" ^
    # --fps
    # 0
    cmd = "videoed extract-video"
    args = {
        "--input-file": input_file,
        "--output-dir": output_dir,
        "--fps": fps,
    }
    dfl_exec(cmd, args)


def get_root_path():
    import os
    path = __file__
    for _ in range(3):
        path = os.path.dirname(path)
    return path


def dfl_exec(cmd, args):
    import subprocess
    s = ""
    s += "@echo off\n"
    s += "call _internal\\setenv.bat\n"
    s += "\"%PYTHON_EXECUTABLE%\" \"%DFL_ROOT%\\main.py\" " + cmd
    for k in args:
        v = args[k]
        if isinstance(v, str):
            v = "\"" + v + "\""
        if len(v) == 2:
            s += " ^\n    %s" % k
        else:
            s += " ^\n    %s %s" % (k, v)
    fpath = os.path.join(get_root_path(), "@exec.bat")
    with open(fpath, "w") as f:
        f.write(s)
    subprocess.call([fpath])


def dfl_load_img(path):
    from pathlib import Path
    from utils.DFLPNG import DFLPNG
    from utils.DFLJPG import DFLJPG
    filepath = Path(path)
    if filepath.suffix == '.png':
        dflimg = DFLPNG.load(str(filepath))
    elif filepath.suffix == '.jpg':
        dflimg = DFLJPG.load(str(filepath))
    else:
        dflimg = None
    if dflimg is None:
        io.log_err("%s is not a dfl image file" % (filepath.name))
    return dflimg


def dfl_estimate_pitch_yaw_roll(dfl_img):
    from facelib import LandmarksProcessor
    return LandmarksProcessor.estimate_pitch_yaw_roll(dfl_img.get_landmarks())
