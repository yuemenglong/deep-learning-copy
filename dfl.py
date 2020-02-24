import os


def dfl_train(src_aligned, dst_aligned, model_dir, model="SAE"):
    cmd = "train"
    args = {
        "--training-data-src-dir": src_aligned,
        "--training-data-dst-dir": dst_aligned,
        "--model-dir": model_dir,
        "--model": model,
        "--pretraining-data-dir": "%INTERNAL%\\pretrain_CelebA",
    }
    dfl_exec(cmd, args)


def dfl_merge(input_dir, output_dir, aligned_dir, model_dir, model="SAE"):
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


def dfl_exec(cmd, args, env=""):
    import subprocess
    s = ""
    s += "@echo off\n"
    s += "call _internal\\setenv%s.bat\n" % env
    s += "\"%PYTHON_EXECUTABLE%\" \"%DFL_ROOT%\\main.py\" " + cmd
    for k in args:
        v = args[k]
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
    fpath = os.path.join(get_root_path(), "@exec.bat")
    with open(fpath, "w") as f:
        f.write(s)
    subprocess.call([fpath])


def dfl_load_img(path):
    from pathlib import Path
    from DFLIMG import DFLPNG
    from DFLIMG import DFLJPG
    filepath = Path(path)
    if filepath.suffix == '.png':
        dflimg = DFLPNG.load(str(filepath))
    elif filepath.suffix == '.jpg':
        dflimg = DFLJPG.load(str(filepath))
    else:
        dflimg = None
    if dflimg is None:
        print("%s is not a dfl image file" % (filepath.name))
    return dflimg


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


def dfl_extract_faces(input_dir, output_dir, detector="s3fd", manual_fix=False):
    # "%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py"
    # extract ^
    # --input - dir
    # "%WORKSPACE%\data_src" ^
    # --output - dir
    # "%WORKSPACE%\data_src\aligned" ^
    # --detector
    # s3fd
    cmd = "extract"
    args = {
        "--input-dir": input_dir,
        "--output-dir": output_dir,
        "--detector": detector,
        "--manual-fix": manual_fix,
    }
    dfl_exec(cmd, args)