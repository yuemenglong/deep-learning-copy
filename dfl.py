import os

_config = {}


def load_config():
    for line in open(get_exec_path()):
        if not line.startswith("rem config "):
            continue
        line = line.split("rem config ")[1]
        key = line.split("=")[0]
        value = line.split("=")[1]
        _config[key] = value


def set_config(key, value):
    _config[key] = value


def get_config(key, dft_value):
    if key in _config:
        return _config[key]
    else:
        return dft_value


def dfl_train(src_aligned, dst_aligned, model_dir, model="SAEHD"):
    cmd = "train"
    args = {
        "--training-data-src-dir": src_aligned,
        "--training-data-dst-dir": dst_aligned,
        "--model-dir": model_dir,
        "--model": model,
        "--pretraining-data-dir": "%INTERNAL%\\pretrain_CelebA",
    }
    dfl_exec(cmd, args)


def dfl_merge(input_dir, output_dir, aligned_dir, model_dir, model="SAEHD"):
    cmd = "merge"
    args = {
        "--input-dir": input_dir,
        "--output-dir": output_dir,
        "--output-mask-dir": output_dir + "_mask",
        "--aligned-dir": aligned_dir,
        "--model-dir": model_dir,
        "--model": model,
    }
    dfl_exec(cmd, args)


def dfl_sort_by_hist(input_dir):
    cmd = "sort"
    args = {
        "--input-dir": input_dir,
        "--by": "hist",
    }
    dfl_exec(cmd, args)


def dfl_sort_by_final(input_dir):
    cmd = "sort"
    args = {
        "--input-dir": input_dir,
        "--by": "final",
    }
    dfl_exec(cmd, args)


def dfl_sort_by_absdiff(input_dir):
    cmd = "sort"
    args = {
        "--input-dir": input_dir,
        "--by": "absdiff",
    }
    dfl_exec(cmd, args)


def dfl_sort_by_vggface(input_dir):
    cmd = "sort"
    args = {
        "--input-dir": input_dir,
        "--by": "vggface",
    }
    dfl_exec(cmd, args)


def dfl_recover_filename(input_dir):
    # "%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py"
    # util ^
    # --input - dir
    # "%WORKSPACE%\data_src\aligned" ^
    # --recover - original - aligned - filename
    cmd = "util"
    args = {
        "--input-dir": input_dir,
        "--recover-original-aligned-filename": "",
    }
    dfl_exec(cmd, args)


def dfl_video_from_sequence(input_dir, output_file, reference_file):
    cmd = "videoed video-from-sequence"
    args = {
        "--input-dir": input_dir,
        "--output-file": output_file,
        "--reference-file": reference_file,
        "--include-audio": True,
    }
    dfl_exec(cmd, args)


def dfl_extract_video(input_file, output_dir, fps=0, output_ext="png"):
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
        "--output-ext": output_ext,
        "--fps": fps,
    }
    dfl_exec(cmd, args)


def dfl_edit_mask(input_dir):
    # "%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py"
    # labelingtool
    # edit_mask ^
    # --input - dir
    # "%WORKSPACE%\data_dst\aligned" ^
    # --confirmed - dir
    # "%WORKSPACE%\data_dst\aligned_confirmed" ^
    # --skipped - dir
    # "%WORKSPACE%\data_dst\aligned_skipped"
    cmd = "labelingtool edit_mask"
    args = {
        "--input-dir": input_dir,
        "--confirmed-dir": input_dir + "_confirmed",
        "--skipped-dir": input_dir + "_skipped",
    }
    dfl_exec(cmd, args)
    return [input_dir, input_dir + "_confirmed", input_dir + "_skipped"]


def get_root_path():
    import os
    path = __file__
    for _ in range(3):
        path = os.path.dirname(path)
    return path


def get_exec_path():
    return os.path.join(get_root_path(), "@exec.bat")


def dfl_exec(cmd, args, env=""):
    import subprocess
    s = ""
    s += "@echo off\n"
    s += "call _internal\\setenv%s.bat\n" % env
    s += "\"%PYTHON_EXECUTABLE%\" \"%DFL_ROOT%\\main.py\" " + cmd
    for k in args:
        v = args[k]
        if v is None:
            continue
        if isinstance(v, str):
            v = "\"" + v + "\""
        # 2相当于只有引号的空字符串
        if isinstance(v, str) and len(v) == 2:
            s += " ^\n    %s" % k
        elif isinstance(v, bool):
            if v:
                s += " ^\n    %s" % k
        else:
            s += " ^\n    %s %s" % (k, v)
    for k in _config:
        s += "\nrem config %s=%s" % (k, _config[k])
    fpath = get_exec_path()
    with open(fpath, "w") as f:
        f.write(s)
    subprocess.call([fpath])


def dfl_img_load(path):
    from pathlib import Path
    from DFLIMG import DFLJPG
    filepath = Path(path)
    if filepath.suffix == '.jpg':
        dflimg = DFLJPG.load(str(filepath))
    else:
        dflimg = None
    if dflimg is None:
        print("%s is not a dfl image file" % (filepath.name))
    return dflimg


def dfl_img_area(dfl_img):
    source_rect = dfl_img.get_source_rect()
    from core import mathlib
    import numpy as np
    rect_area = mathlib.polygon_area(np.array(source_rect[[0, 2, 2, 0]]).astype(np.float32),
                                     np.array(source_rect[[1, 1, 3, 3]]).astype(np.float32))
    return rect_area


def dfl_estimate_pitch_yaw_roll(dfl_img):
    from facelib import LandmarksProcessor
    return LandmarksProcessor.estimate_pitch_yaw_roll(dfl_img.get_landmarks())


def dfl_faceset_metadata_save(input_dir):
    # "%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py"
    # util ^
    # --input - dir
    # "%WORKSPACE%\data_src\aligned" ^
    # --save - faceset - metadata
    cmd = "util"
    args = {
        "--input-dir": input_dir,
        "--save-faceset-metadata": "",
    }
    dfl_exec(cmd, args)


def dfl_faceset_metadata_restore(input_dir):
    # "%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py"
    # util ^
    # --input - dir
    # "%WORKSPACE%\data_src\aligned" ^
    # --save - faceset - metadata
    cmd = "util"
    args = {
        "--input-dir": input_dir,
        "--restore-faceset-metadata": "",
    }
    dfl_exec(cmd, args)


def dfl_extract_faces(input_dir, output_dir, detector="s3fd", manual_fix=False, output_debug=False):
    cmd = "extract"
    args = {
        "--input-dir": input_dir,
        "--output-dir": output_dir,
        # "--output-debug": output_debug,
        "--manual-fix": manual_fix,
        "--detector": detector,
        "--face-type": "whole_face",
        "--max-faces-from-image": 3,
        "--image-size": 512,
        "--jpeg-quality": 90,
        # p.add_argument('--output-debug', action="store_true", dest="output_debug", default=None, help="Writes debug images to <output-dir>_debug\ directory.")
        # p.add_argument('--no-output-debug', action="store_false", dest="output_debug", default=None, help="Don't writes debug images to <output-dir>_debug\ directory.")
        # p.add_argument('--face-type', dest="face_type", choices=['half_face', 'full_face', 'whole_face', 'head', 'mark_only'], default=None)
        # p.add_argument('--max-faces-from-image', type=int, dest="max_faces_from_image", default=None, help="Max faces from image.")
        # p.add_argument('--image-size', type=int, dest="image_size", default=None, help="Output image size.")
        # p.add_argument('--jpeg-quality', type=int, dest="jpeg_quality", default=None, help="Jpeg quality.")
    }
    if output_debug:
        args["--output-debug"] = True
    else:
        args["--no-output-debug"] = True
    dfl_exec(cmd, args)


def dfl_xseg_editor(input_dir):
    cmd = "xseg editor"
    args = {
        "--input-dir": input_dir,
    }
    dfl_exec(cmd, args)


def dfl_xseg_fetch(input_dir):
    cmd = "xseg fetch"
    args = {
        "--input-dir": input_dir,
    }
    dfl_exec(cmd, args)


def dfl_xseg_train(src_aligned, dst_aligned, model_dir):
    dfl_train(src_aligned, dst_aligned, model_dir, "XSeg")
