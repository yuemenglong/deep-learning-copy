import numpy as np
import os


def print_stack():
    import traceback
    for line in traceback.format_stack():
        print(line.strip())


def mid_point(points):
    x = sum(p[0] for p in points) / len(points)
    y = sum(p[1] for p in points) / len(points)
    return x, y


def poly_area(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def angle_between(v1, v2):
    def unit_vector(vector):
        return vector / np.linalg.norm(vector)

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def skip_no_face(dir, pat="%05d"):
    import os
    import shutil
    import sys
    aligend_dir = os.path.join(dir, "aligned")
    aligend = set([f.split("_")[0] for f in os.listdir(aligend_dir)])
    merged_dir = os.path.join(dir, "merged")
    merged_dir_bak = os.path.join(dir, "merged_bak")
    if os.path.exists(merged_dir_bak):
        raise Exception("Merge Dir Bak Exists")
    shutil.move(merged_dir, merged_dir_bak)
    os.mkdir(merged_dir)
    idx = 0
    for f in os.listdir(merged_dir_bak):
        name = os.path.splitext(f)[0]
        ext = os.path.splitext(f)[-1]
        if name in aligend:
            idx += 1
            sys.stdout.write("Skip No Face Proc: %d\r" % idx)
            sys.stdout.flush()
            src = os.path.join(merged_dir_bak, f)
            dst = os.path.join(merged_dir, "%05d" % idx + ext)
            shutil.move(src, dst)


def cpu_count():
    from multiprocessing import cpu_count as cs
    return cs()


def get_root_path():
    import os
    path = __file__
    for _ in range(3):
        path = os.path.dirname(path)
    return path


def get_time_str():
    import time
    return time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))


def backup_model(model_name, model_path):
    import os
    import shutil
    backup_path = os.path.join(model_path, "backup")
    if not os.path.exists(backup_path):
        os.mkdir(backup_path)
    for file in os.listdir(model_path):
        if os.path.isdir(os.path.join(model_path, file)):
            continue
        if file.startswith(model_name):
            src = os.path.join(model_path, file)
            dst = os.path.join(backup_path, file)
            shutil.copy(src, dst)


def has_backup(model_name, model_path):
    import os
    backup_path = os.path.join(model_path, "backup")
    if not os.path.exists(backup_path):
        return False
    for file in os.listdir(model_path):
        if os.path.isdir(os.path.join(model_path, file)):
            continue
        if file.startswith(model_name):
            return True
    return False


def restore_model(model_name, model_path):
    import os
    import shutil
    backup_path = os.path.join(model_path, "backup")
    if not os.path.exists(backup_path):
        return
    for file in os.listdir(backup_path):
        if os.path.isdir(os.path.join(model_path, file)):
            continue
        if file.startswith(model_name):
            src = os.path.join(backup_path, file)
            dst = os.path.join(model_path, file)
            shutil.copy(src, dst)


def extract():
    import os
    import shutil
    from mainscripts import VideoEd
    from mainscripts import Extractor
    from interact import interact as io

    root_dir = get_root_path()
    extract_workspace = os.path.join(root_dir, "extract_workspace")
    target_dir = os.path.join(extract_workspace, "aligned_")

    valid_exts = [".mp4", ".avi", ".wmv", ".mkv"]

    fps = io.input_int("Enter FPS ( ?:help skip:fullfps ) : ", 0,
                       help_message="How many frames of every second of the video will be extracted.")

    def file_filter(file):
        if os.path.isdir(os.path.join(extract_workspace, file)):
            return False
        ext = os.path.splitext(file)[-1]
        if ext not in valid_exts:
            return False
        return True

    files = list(filter(file_filter, os.listdir(extract_workspace)))
    files.sort()
    pos = 0
    for file in files:
        pos += 1
        print("@@@@@  Start Process " + file, "%d / %d" % (pos, len(files)))
        # 提取图片
        input_file = os.path.join(extract_workspace, file)
        output_dir = os.path.join(extract_workspace, "extract_images")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))
        VideoEd.extract_video(input_file, output_dir, output_ext="png", fps=fps)
        print("@@@@@  Start Extract " + file, "%d / %d" % (pos, len(files)))
        # 提取人脸
        input_dir = output_dir
        output_dir = os.path.join(extract_workspace, "_current")
        debug_dir = os.path.join(extract_workspace, "debug")
        min_pixel = 512 if fps % 5 == 0 and fps != 0 else 256
        Extractor.main(input_dir, output_dir, debug_dir, "s3fd", min_pixel=min_pixel)
        # fanseg
        print("@@@@@  Start FanSeg " + file, "%d / %d" % (pos, len(files)))
        Extractor.extract_fanseg(output_dir)
        # 复制到结果集
        print("@@@@@  Start Move " + file, "%d / %d" % (pos, len(files)))
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        ts = get_time_str()
        for f in os.listdir(output_dir):
            src = os.path.join(output_dir, f)
            dst = os.path.join(target_dir, "%s_%s" % (ts, f))
            shutil.move(src, dst)
        # 全部做完，删除该文件
        print("@@@@@  Finish " + file, "%d / %d" % (pos, len(files)))
        os.remove(os.path.join(extract_workspace, file))
        os.rmdir(output_dir)


# noinspection PyUnresolvedReferences
def sort_by_hist(input_path):
    import mainscripts.Sorter as Sorter
    Sorter.sort_by_hist(input_path)


# noinspection PyUnresolvedReferences
def sort_by_pitch(input_path):
    import mainscripts.Sorter as Sorter
    Sorter.sort_by_face_pitch(input_path)


# noinspection PyUnresolvedReferences
def get_pitch_yaw_roll(input_path):
    import os
    import numpy as np
    import cv2
    from shutil import copyfile
    from pathlib import Path
    from utils import Path_utils
    from utils.DFLPNG import DFLPNG
    from utils.DFLJPG import DFLJPG
    from facelib import LandmarksProcessor
    from joblib import Subprocessor
    import multiprocessing
    from interact import interact as io
    from imagelib import estimate_sharpness
    io.log_info("Sorting by face yaw...")
    img_list = []
    trash_img_list = []
    for filepath in io.progress_bar_generator(Path_utils.get_image_paths(input_path), "Loading"):
        filepath = Path(filepath)
        if filepath.suffix == '.png':
            dflimg = DFLPNG.load(str(filepath))
        elif filepath.suffix == '.jpg':
            dflimg = DFLJPG.load(str(filepath))
        else:
            dflimg = None
        if dflimg is None:
            io.log_err("%s is not a dfl image file" % (filepath.name))
            trash_img_list.append([str(filepath)])
            continue
        pitch, yaw, roll = LandmarksProcessor.estimate_pitch_yaw_roll(dflimg.get_landmarks())
        img_list.append([str(filepath), pitch, yaw, roll])

    img_list.sort(key=lambda item: item[1])
    with open(os.path.join(input_path, "_pitch_yaw_roll.csv"), "w") as f:
        for i in img_list:
            f.write("%s,%f,%f,%f\n" % (i[0], i[1], i[2], i[3]))

    import cv
    width = 800
    trans = lambda x: int((x + 1) * width / 2)
    img = cv.cv_new((width, width))
    min = trans(-1)
    max = trans(1)
    # points
    for l in img_list:
        pitch = trans(l[1])
        yaw = trans(l[2])
        cv.cv_point(img, (pitch, yaw), (0xcc, 0x66, 0x33), 2)
    # border
    for i in range(-10, 10, 2):
        x = trans(i / 10)
        thick = 1
        if i % 4 == 0:
            thick = 2
        if i == 0:
            thick = 3
        cv.cv_line(img, (min, x), (max, x), (0, 0, 0), thick)
        cv.cv_line(img, (x, min), (x, max), (0, 0, 0), thick)
    cv.cv_save(img, os.path.join(input_path, "_pitch_yaw_roll.bmp"))
    import shutil
    shutil.copy(os.path.join(input_path, "_pitch_yaw_roll.csv"), "C:/users/yml/desktop")
    shutil.copy(os.path.join(input_path, "_pitch_yaw_roll.bmp"), "C:/users/yml/desktop")
    return img_list


def get_image_var(img_path):
    import cv2
    image = cv2.imread(img_path)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return image_var


def show_landmarks(path):
    import cv
    from utils.DFLJPG import DFLJPG
    jpg = DFLJPG.load(path)
    img = cv.cv_load(path)
    lm = jpg.get_landmarks()
    for (x, y) in lm:
        cv.cv_point(img, (x, y), (255, 0, 0))
    cv.cv_show(img)


def get_extract_pitch_yaw_roll():
    get_pitch_yaw_roll(os.path.join(get_root_path(), "extract_workspace", "aligned_"))


def get_data_src_pitch_yaw_roll():
    get_pitch_yaw_roll(os.path.join(get_root_path(), "workspace", "data_src", "aligned"))


def get_data_dst_pitch_yaw_roll():
    get_pitch_yaw_roll(os.path.join(get_root_path(), "workspace", "data_dst", "aligned"))


def skip_spec_pitch(input_path):
    import os
    img_list = get_pitch_yaw_roll(input_path)
    for [path, pitch, _, _] in img_list:
        if pitch > 0.2:
            print(path)
            os.remove(path)


def pick_spec_pitch(input_path, output_path):
    import shutil
    import os
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    img_list = get_pitch_yaw_roll(input_path)
    for [path, pitch, _yaw, _roll] in img_list:
        if pitch > 0:
            print(path)
            shutil.copy(path, output_path)


def main():
    import sys

    arg = sys.argv[-1]
    if arg == '--skip-no-face':
        skip_no_face(os.path.join(get_root_path(), "workspace", "data_dst"))
    elif arg == '--extract':
        extract()
    elif arg == '--skip-spec-pitch':
        skip_spec_pitch(os.path.join(get_root_path(), "workspace", "data_dst", "aligned"))
    else:
        # get_data_src_pitch_yaw_roll()
        # get_data_dst_pitch_yaw_roll()
        # get_extract_pitch_yaw_roll()
        get_pitch_yaw_roll(os.path.join(get_root_path(), "extract_workspace", "aligned_ym_4k_01_16"))
        # pick_spec_pitch(os.path.join(get_root_path(), "extract_workspace/aligned_"),
        #                 os.path.join(get_root_path(), "extract_workspace/ym_bili_pick"))
        pass


if __name__ == '__main__':
    main()
