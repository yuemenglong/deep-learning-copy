import os
from typing import Any, Callable

import numpy as np

from interact import interact as io


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


def skip_no_face(data_dst_dir):
    import os
    import shutil
    data_dst_aligned_dir = os.path.join(data_dst_dir, "aligned")
    aligend = set([f.split(".")[0].split("_")[0] for f in os.listdir(data_dst_aligned_dir)])
    merged_dir = os.path.join(data_dst_dir, "merged")
    merged_trash_dir = os.path.join(data_dst_dir, "merged_trash")
    if os.path.exists(merged_trash_dir):
        # raise Exception("Merge Dir Bak Exists")
        shutil.rmtree(merged_trash_dir)
    shutil.move(merged_dir, merged_trash_dir)
    os.mkdir(merged_dir)
    idx = 0
    for f in io.progress_bar_generator(os.listdir(merged_trash_dir), "Skip No Face"):
        name = os.path.splitext(f)[0]
        ext = os.path.splitext(f)[-1]
        if name in aligend:
            idx += 1
            src = os.path.join(merged_trash_dir, f)
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
        io.log_info("@@@@@  Start Process %s, %d / %d" % (file, pos, len(files)))
        # 提取图片
        input_file = os.path.join(extract_workspace, file)
        output_dir = os.path.join(extract_workspace, "extract_images")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))
        VideoEd.extract_video(input_file, output_dir, output_ext="png", fps=fps)
        io.log_info("@@@@@  Start Extract %s, %d / %d" % (file, pos, len(files)))
        # 提取人脸
        input_dir = output_dir
        output_dir = os.path.join(extract_workspace, "_current")
        debug_dir = os.path.join(extract_workspace, "debug")
        min_pixel = 512 if fps % 5 == 0 and fps != 0 else 256
        Extractor.main(input_dir, output_dir, debug_dir, "s3fd", min_pixel=min_pixel)
        # fanseg
        io.log_info("@@@@@  Start FanSeg %s, %d / %d" % (file, pos, len(files)))
        Extractor.extract_fanseg(output_dir)
        # 复制到结果集
        io.log_info("@@@@@  Start Move %s, %d / %d" % (file, pos, len(files)))
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        ts = get_time_str()
        for f in os.listdir(output_dir):
            src = os.path.join(output_dir, f)
            dst = os.path.join(target_dir, "%s_%s" % (ts, f))
            shutil.move(src, dst)
        # 全部做完，删除该文件
        io.log_info("@@@@@  Finish %s, %d / %d" % (file, pos, len(files)))
        os.remove(os.path.join(extract_workspace, file))
        os.rmdir(output_dir)


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
    # shutil.copy(os.path.join(input_path, "_pitch_yaw_roll.csv"), get_desktop_path())
    # shutil.copy(os.path.join(input_path, "_pitch_yaw_roll.bmp"), get_desktop_path())
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


def get_desktop_path():
    return "C:/users/yml/desktop"


def csv_name():
    return "_pitch_yaw_roll.csv"


def skip_by_pitch(src_path, dst_path):
    import os
    import shutil
    import cv
    # src_csv = os.path.join(src_path, csv_name())
    # dst_csv = os.path.join(dst_path, csv_name())
    # if not os.path.exists(src_csv):
    #     get_pitch_yaw_roll(src_path)
    # if not os.path.exists(dst_csv):
    #     get_pitch_yaw_roll(dst_path)
    src_img_list = get_pitch_yaw_roll(src_path)
    dst_img_list = get_pitch_yaw_roll(dst_path)
    # with open(src_csv) as f:
    #     for line in f.readlines():
    #         [path, pitch, yaw, roll] = line.strip().split(",")
    #         src_img_list.append([path, float(pitch), float(yaw), float(roll)])
    # with open(dst_csv) as f:
    #     for line in f.readlines():
    #         [path, pitch, yaw, roll] = line.strip().split(",")
    #         dst_img_list.append([path, float(pitch), float(yaw), float(roll)])
    trash_path = dst_path + "_trash"
    if not os.path.exists(trash_path):
        os.makedirs(trash_path)
    size = 800
    r = 20
    img = cv.cv_new((size + 1, size + 1))
    trans: Callable[[Any], int] = lambda v: int((v + 1) * size / 2)
    count = 0
    for [_, pitch, yaw, _] in src_img_list:
        x = trans(pitch)
        y = trans(yaw)
        cv.cv_point(img, (x, y), (128, 128, 128), r)
    # cv.cv_show(img)
    xys = []
    for [path, pitch, yaw, _] in dst_img_list:
        x = trans(pitch)
        y = trans(yaw)
        c = img[y, x]
        if sum(c) == 255 * 3:
            xys.append((x, y, (0, 0, 0xff)))
            if not os.path.exists(path) or not os.path.exists(trash_path):
                continue
            count += 1
            shutil.move(path, trash_path)
        else:
            xys.append((x, y, (0xcc, 0x66, 0x33)))
    for (x, y, color) in xys:
        cv.cv_point(img, (x, y), color, 2)
    # border
    delta = int(size / 10)
    for i in range(0, size, delta):
        x = i
        thick = 1
        if i in [delta, 3 * delta, 7 * delta, 9 * delta]:
            thick = 2
        if i == delta * 5:
            thick = 3
        cv.cv_line(img, (0, x), (size, x), (0, 0, 0), thick)
        cv.cv_line(img, (x, 0), (x, size), (0, 0, 0), thick)
    # cv.cv_show(img)
    io.log_info("Out Of Pitch, %d / %d" % (len(dst_img_list), count))
    # save_path = os.path.join(get_desktop_path(), "skip_by_pitch.bmp")
    # cv.cv_save(img, save_path)


def split_aligned():
    import os
    import shutil
    aligned_path = os.path.join(get_root_path(), "extract_workspace", "aligned_")
    count = 0
    dst_dir = os.path.join(get_root_path(), "extract_workspace", "split_%02d" % int(count / 10000))
    for f in os.listdir(aligned_path):
        if not f.endswith(".jpg") and not f.endswith(".png"):
            continue
        if count % 10000 == 0:
            print(count)
            dst_dir = os.path.join(get_root_path(), "extract_workspace", "split_%02d" % int(count / 10000))
            os.mkdir(dst_dir)
        src = os.path.join(aligned_path, f)
        shutil.move(src, dst_dir)
        count += 1


def match_by_pitch(data_src_path, data_dst_path, output_path=None):
    r = 0.05
    mn = 1
    mx = 2
    import cv
    import shutil
    # 准备各种路径
    src_aligned_store = os.path.join(data_src_path, "aligned_store")
    if not os.path.exists(src_aligned_store):
        raise Exception("No Src Aligned Store")
    src_aligned = output_path if output_path is not None else os.path.join(data_src_path, "aligned")
    if os.path.exists(src_aligned):
        shutil.rmtree(src_aligned)
    os.mkdir(src_aligned)
    dst_aligned = os.path.join(data_dst_path, "aligned")
    dst_aligned_trash = os.path.join(data_dst_path, "aligned_trash")
    if not os.path.exists(dst_aligned_trash):
        os.mkdir(dst_aligned_trash)
    # 读取角度信息
    src_img_list = get_pitch_yaw_roll(src_aligned_store)
    dst_img_list = get_pitch_yaw_roll(dst_aligned)
    src_pitch = list([i[1] for i in src_img_list])
    src_yaw = list([i[2] for i in src_img_list])
    dst_pitch = list([i[1] for i in dst_img_list])
    dst_yaw = list([i[2] for i in dst_img_list])
    src_ps = np.array(list(zip(src_pitch, src_yaw)), "float")
    dst_ps = np.array(list(zip(dst_pitch, dst_yaw)), "float")

    # 计算最近的n个点
    src_match = set()
    dst_match = set()
    for p, i in io.progress_bar_generator(zip(dst_ps, range(len(dst_ps))), "Calculating"):
        ds = np.linalg.norm(src_ps - p, axis=1, keepdims=True)
        idxs = np.argsort(ds, axis=0)
        min_idx = idxs[mn - 1][0]
        # 极端情况所有距离都不满足半径范围
        if ds[min_idx] > r:
            continue
        # 至少有一个满足半径条件了,dst_point可以留下
        dst_match.add(i)
        # 所有满足条件的加入到src_match
        for idx in idxs[:mx]:
            idx = idx[0]
            if ds[idx] > r:
                break
            src_match.add(idx)
    if not os.path.exists(src_aligned):
        os.mkdir(src_aligned)
    io.log_info("%s, %s, %s, %s" % ("Src Match", len(src_match), "Src All", len(src_img_list)))
    io.log_info("%s, %s, %s, %s" % ("Dst Match", len(dst_match), "Dst All", len(dst_img_list)))

    # 画图
    xycr = []
    for idx in range(len(src_img_list)):
        t = src_img_list[idx]
        if idx in src_match:
            xycr.append([t[1], t[2], (128, 128, 128), int(r * 400)])  # 蓝色，匹配到的
            shutil.copy(t[0], src_aligned)
        else:
            xycr.append([t[1], t[2], (128, 128, 128), 2])  # 灰色，没匹配到
    for idx in range(len(dst_img_list)):
        t = dst_img_list[idx]
        if idx in dst_match:
            xycr.append([t[1], t[2], (0, 255, 0), 2])  # 绿色，保留
        else:
            xycr.append([t[1], t[2], (0, 0, 255), 2])  # 红色，删除
            shutil.move(t[0], dst_aligned_trash)
    img = cv.cv_new((800 + 1, 800 + 1))
    xs = [i[0] for i in xycr]
    ys = [i[1] for i in xycr]
    cs = [i[2] for i in xycr]
    rs = [i[3] for i in xycr]
    cv.cv_scatter(img, xs, ys, [-1, 1], [-1, 1], cs, rs)
    cv.cv_save(img, os.path.join(dst_aligned, "_match_by_pitch.bmp"))


# noinspection PyUnresolvedReferences
def sort_by_hist(input_path):
    import mainscripts.Sorter as Sorter
    img_list = Sorter.sort_by_hist(input_path)
    Sorter.final_process(input_path, img_list, [])


# noinspection PyUnresolvedReferences
def sort_by_origname(input_path):
    import mainscripts.Sorter as Sorter
    img_list, _ = Sorter.sort_by_origname(input_path)
    Sorter.final_process(input_path, img_list, [])


def prepare(workspace):
    import os
    import shutil
    from mainscripts import Extractor
    from mainscripts import VideoEd
    for f in os.listdir(workspace):
        ext = os.path.splitext(f)[-1]
        if ext not in ['.mp4', '.avi']:
            continue
        if f.startswith("result"):
            continue
        # 获取所有的data_dst文件
        tmp_dir = os.path.join(workspace, "_tmp")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        video = os.path.join(workspace, f)
        # 提取帧
        VideoEd.extract_video(video, tmp_dir, "png", 0)
        # 提取人脸
        Extractor.main(tmp_dir, os.path.join(tmp_dir, "aligned"), detector='s3fd')
        # 两组人脸匹配
        match_by_pitch(os.path.join(workspace, "data_src"), tmp_dir, os.path.join(tmp_dir, "src"))
        # 排序
        sort_by_hist(os.path.join(tmp_dir, "aligned"))
        # 重命名
        fname = f.replace(ext, "")
        dst_dir = os.path.join(workspace, "data_dst_%s_%s" % (get_time_str(), fname))
        shutil.move(tmp_dir, dst_dir)
        data_trash = os.path.join(workspace, "data_trash")
        if not os.path.exists(data_trash):
            os.mkdir(data_trash)
        shutil.move(video, data_trash)


def train(workspace, target_loss=0.12):
    import os
    from mainscripts import Trainer
    model_path = os.path.join(workspace, "model")
    train_args = {
        'training_data_src_dir': '',
        'training_data_dst_dir': '',
        'pretraining_data_dir': None,
        'model_path': model_path,
        'model_name': 'SAE',
        'no_preview': False,
        'debug': False,
        'execute_programs': [],
        'target_loss': target_loss
    }
    device_args = {'cpu_only': False, 'force_gpu_idx': -1}
    for f in os.listdir(workspace):
        if not os.path.isdir(os.path.join(workspace, f)) or not f.startswith("data_dst_"):
            continue
        io.log_info(f)
        data_dst = os.path.join(workspace, f)
        data_src_aligned = os.path.join(data_dst, "src")
        data_dst_aligned = os.path.join(data_dst, "aligned")
        # 训练
        train_args['training_data_src_dir'] = data_src_aligned
        train_args['training_data_dst_dir'] = data_dst_aligned
        Trainer.main(train_args, device_args)
        return


def convert(workspace):
    import os
    from mainscripts import Converter
    from mainscripts import VideoEd
    from converters import ConverterMasked
    convert_args = {
        'input_dir': 'D:\\DeepFaceLabCUDA10.1AVX\\workspace\\data_dst',
        'output_dir': 'D:\\DeepFaceLabCUDA10.1AVX\\workspace\\data_dst\\merged',
        'aligned_dir': 'D:\\DeepFaceLabCUDA10.1AVX\\workspace\\data_dst\\aligned',
        'avaperator_aligned_dir': None,
        'model_dir': 'D:\\DeepFaceLabCUDA10.1AVX\\workspace\\model',
        'model_name': 'SAE',
        'debug': False
    }
    device_args = {'cpu_only': False, 'force_gpu_idx': -1}
    for f in os.listdir(workspace):
        if not os.path.isdir(os.path.join(workspace, f)) or not f.startswith("data_dst_"):
            continue
        io.log_info(f)
        data_dst = os.path.join(workspace, f)
        data_dst_merged = os.path.join(data_dst, "merged")
        data_dst_aligned = os.path.join(data_dst, "aligned")
        # 恢复排序
        sort_by_origname(data_dst_aligned)
        # 转换
        convert_args['input_dir'] = data_dst
        convert_args['output_dir'] = data_dst_merged
        convert_args['aligned_dir'] = data_dst_aligned
        ConverterMasked.enable_predef = True
        Converter.main(convert_args, device_args)
        # 去掉没有脸的
        skip_no_face(data_dst)
        # 合成
        refer_name = "_".join(f.split("_")[8:])
        refer_path = None
        result_path = os.path.join(workspace, "result.mp4")
        data_trash = os.path.join(workspace, "data_trash")
        for ff in os.listdir(data_trash):
            if ff.startswith(refer_name):
                refer_path = os.path.join(data_trash, ff)
                break
        VideoEd.video_from_sequence(data_dst_merged, result_path, refer_path, "png", None, None, False)
        return


def next(workspace):
    import shutil
    for f in os.listdir(workspace):
        if os.path.isdir(os.path.join(workspace, f)) and f.startswith("data_dst"):
            src = os.path.join(workspace, f)
            dst = os.path.join(workspace, "data_trash")
            io.log_info("Move %s To %s" % (src, dst))
            shutil.move(src, dst)
            return


def main():
    import sys

    arg = sys.argv[-1]
    if arg == '--skip-no-face':
        skip_no_face(os.path.join(get_root_path(), "workspace", "data_dst"))
    elif arg == '--extract':
        extract()
    elif arg == '--skip-by-pitch':
        skip_by_pitch(os.path.join(get_root_path(), "workspace/data_src/aligned"),
                      os.path.join(get_root_path(), "workspace/data_dst/aligned"))
    elif arg == '--split-aligned':
        split_aligned()
    elif arg == '--prepare':
        prepare(os.path.join(get_root_path(), "workspace"))
    elif arg == '--train':
        train(os.path.join(get_root_path(), "workspace"))
    elif arg == '--convert':
        convert(os.path.join(get_root_path(), "workspace"))
    elif arg == '--next':
        next(os.path.join(get_root_path(), "workspace"))
    else:
        for f in os.listdir(os.path.join(get_root_path(), "workspace")):
            if os.path.isdir(os.path.join(get_root_path(), "workspace", f)) and f.startswith("data_dst_"):
                print(f)
                match_by_pitch(os.path.join(get_root_path(), "workspace", "data_src"),
                               os.path.join(get_root_path(), "workspace", f),
                               os.path.join(get_root_path(), "workspace", f, "src"))
        pass
        # prepare_train(os.path.join(get_root_path(), "workspace"))
        # train(os.path.join(get_root_path(), "workspace"))
        # sort_by_hist(os.path.join(get_root_path(), "workspace/data_src/aligned"))
        # match_by_pitch(os.path.join(get_root_path(), "workspace/data_src"),
        #                os.path.join(get_root_path(), "workspace/data_dst")
        #                )
        # split_aligned()
        # skip_by_pitch(os.path.join(get_root_path(), "workspace/data_src/aligned"),
        #               os.path.join(get_root_path(), "workspace/data_dst/aligned"))
        # get_data_src_pitch_yaw_roll()
        # get_data_dst_pitch_yaw_roll()
        # get_extract_pitch_yaw_roll()
        # get_pitch_yaw_roll(os.path.join(get_root_path(), "extract_workspace", "aligned_ab_01_30"))
        # pick_spec_pitch(os.path.join(get_root_path(), "extract_workspace/aligned_"),
        #                 os.path.join(get_root_path(), "extract_workspace/ym_bili_pick"))
        pass


if __name__ == '__main__':
    main()
