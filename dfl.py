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
    exec(cmd, args)


def dfl_convert(input_dir, output_dir, aligned_dir, model_dir, model="SAE"):
    cmd = "convert"
    args = {
        "--input-dir": input_dir,
        "--output-dir": output_dir,
        "--aligned-dir": aligned_dir,
        "--model-dir": model_dir,
        "--model": model,
    }
    exec(cmd, args)


def dfl_video_from_sequence(input_dir, output_file, reference_file):
    cmd = "videoed video-from-sequence"
    args = {
        "--input-dir": input_dir,
        "--output-file": output_file,
        "--reference-file": reference_file,
    }
    exec(cmd, args)


def get_root_path():
    import os
    path = __file__
    for _ in range(3):
        path = os.path.dirname(path)
    return path


def exec(cmd, args):
    import subprocess
    s = ""
    s += "@echo off\n"
    s += "call _internal\\setenv.bat\n"
    s += "\"%PYTHON_EXECUTABLE%\" \"%DFL_ROOT%\\main.py\" " + cmd
    for k in args:
        v = args[k]
        if isinstance(v, str):
            v = "\"" + v + "\""
        s += " ^\n    %s %s" % (k, v)
    fpath = os.path.join(get_root_path(), "@exec.bat")
    with open(fpath, "w") as f:
        f.write(s)
    subprocess.call([fpath])
