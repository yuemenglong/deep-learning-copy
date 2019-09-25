import os
import dfl
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
    aligend = set([f.split('_')[0] for f in os.listdir(data_dst_aligned_dir)])
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
            shutil.copy(src, dst)


def cpu_count():
    from multiprocessing import cpu_count as cs
    return cs()


def get_root_path():
    return dfl.get_root_path()


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


def backup_model_move(model_name, model_path):
    import os
    import shutil
    backup_path = os.path.join(model_path, "backup")
    if not os.path.exists(backup_path):
        return
    move_path = os.path.join(model_path, "backup_move")
    if os.path.exists(move_path):
        shutil.rmtree(move_path)
    os.mkdir(move_path)
    for file in os.listdir(backup_path):
        if os.path.isdir(os.path.join(model_path, file)):
            continue
        if file.startswith(model_name):
            src = os.path.join(backup_path, file)
            dst = os.path.join(move_path, file)
            shutil.move(src, dst)


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

    valid_exts = [".mp4", ".avi", ".wmv", ".mkv", ".ts"]

    fps = io.input_int("Enter FPS ( ?:help skip:fullfps ) : ", 0,
                       help_message="How many frames of every second of the video will be extracted.")
    min_pixel = io.input_int("Enter Min Pixel ( ?:help skip: 512) : ", 512,
                             help_message="Min Pixel")

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


def extract_dst_image(workspace):
    import os
    from mainscripts import Extractor

    # 提取人脸
    input_dir = os.path.join(workspace, "data_dst")
    output_dir = os.path.join(workspace, "data_dst/aligned")
    debug_dir = os.path.join(workspace, "data_dst/aligned_debug")
    Extractor.main(input_dir, output_dir, debug_dir, "s3fd", manual_fix=True)
    # fanseg
    Extractor.extract_fanseg(output_dir)


# noinspection PyUnresolvedReferences
def get_pitch_yaw_roll(input_path, r=0.05):
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
            f.write("%s,%f,%f,%f\n" % (os.path.basename(i[0]), i[1], i[2], i[3]))

    import cv
    width = 800
    img = cv.cv_new((width, width))
    xs = [i[1] for i in img_list]
    ys = [i[2] for i in img_list]
    cs = [(128, 128, 128)] * len(xs)
    rs = [int(r * width / 2)] * len(xs)
    cv.cv_scatter(img, xs, ys, [-1, 1], [-1, 1], cs, rs)
    cs = [(0xcc, 0x66, 0x33)] * len(xs)
    rs = [2] * len(xs)
    cv.cv_scatter(img, xs, ys, [-1, 1], [-1, 1], cs, rs)
    cv.cv_save(img, os.path.join(input_path, "_pitch_yaw_roll.bmp"))
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
    size = 800
    r = 20
    src_img_list = get_pitch_yaw_roll(src_path)
    dst_img_list = get_pitch_yaw_roll(dst_path)
    trash_path = dst_path + "_trash"
    if not os.path.exists(trash_path):
        os.makedirs(trash_path)

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
        c_ = img[-y, x]
        if sum(c) == 255 * 3 and sum(c_) == 255 * 3:
            xys.append((x, y, (0, 0, 0xff)))
            if not os.path.exists(path) or not os.path.exists(trash_path):
                continue
            count += 1
            shutil.move(path, trash_path)
        else:
            xys.append((x, y, (0xcc, 0x66, 0x33)))
    for (x, y, color) in xys:
        cv.cv_point(img, (x, y), color, 2)
    # cv.cv_show(img)
    io.log_info("Out Of Pitch, %d / %d" % (count, len(dst_img_list)))
    save_path = os.path.join(dst_path, "_skip_by_pitch.bmp")
    cv.cv_save(img, save_path)


def split(input_path, target_path, batch=3000):
    import os
    import shutil
    count = 0
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    dst_dir = os.path.join(target_path, "split_%03d" % int(count / batch))
    for f in io.progress_bar_generator(os.listdir(input_path), "Process"):
        if not f.endswith(".jpg") and not f.endswith(".png"):
            continue
        if count % batch == 0:
            dst_dir = os.path.join(target_path, "split_%03d" % int(count / batch))
            os.mkdir(dst_dir)
        src = os.path.join(input_path, f)
        shutil.move(src, dst_dir)
        count += 1


def merge(input_path, target_path):
    import os
    import shutil
    for f in os.listdir(input_path):
        sub_path = os.path.join(input_path, f)
        if os.path.abspath(sub_path) == os.path.abspath(target_path):
            continue
        if os.path.isdir(sub_path):
            time_str = get_time_str()
            for img in io.progress_bar_generator(os.listdir(sub_path), f):
                if img.endswith(".png") or img.endswith(".jpg"):
                    img_path = os.path.join(sub_path, img)
                    dst_path = os.path.join(target_path, "%s_%s" % (time_str, img))
                    shutil.move(img_path, dst_path)


def match_by_pitch(data_src_path, data_dst_path):
    r = 0.05
    mn = 1
    mx = 3
    import cv
    import shutil
    # 准备各种路径
    src_aligned_store = os.path.join(data_src_path, "aligned_store")
    if not os.path.exists(src_aligned_store):
        raise Exception("No Src Aligned Store")
    src_aligned = os.path.join(data_dst_path, "src")
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
    io.log_info("%s, %s, %s, %s" % ("Src Match", len(src_match), "Src All", len(src_img_list)))
    io.log_info("%s, %s, %s, %s" % ("Dst Match", len(dst_match), "Dst All", len(dst_img_list)))

    # 画图
    width = 800
    xycr = []
    for idx in range(len(src_img_list)):
        t = src_img_list[idx]
        if idx in src_match:
            xycr.append([t[1], t[2], (128, 128, 128), int(r * width / 2)])  # 蓝色，匹配到的
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
    img = cv.cv_new((width, width))
    xs = [i[0] for i in xycr]
    ys = [i[1] for i in xycr]
    cs = [i[2] for i in xycr]
    rs = [i[3] for i in xycr]
    cv.cv_scatter(img, xs, ys, [-1, 1], [-1, 1], cs, rs)
    cv.cv_save(img, os.path.join(dst_aligned, "_match_by_pitch.bmp"))

    # 加入base
    base_dir = os.path.join(data_src_path, "aligned_base")
    if os.path.exists(base_dir):
        for img in os.listdir(base_dir):
            if img.endswith(".jpg") or img.endswith(".png"):
                img_path = os.path.join(base_dir, img)
                shutil.copy(img_path, src_aligned)


# noinspection PyUnresolvedReferences
def recover_filename(input_path):
    from mainscripts import Util
    Util.recover_original_aligned_filename(input_path)


def low_prio():
    from utils import os_utils
    os_utils.set_process_lowest_prio()


def manual_select(input_path, src_path=None):
    import cv
    import colorsys
    import cv2
    img_list = []
    src_img_list = []
    width = 800
    ratio = 0.8

    for f in io.progress_bar_generator(os.listdir(input_path), "Loading"):
        if f.endswith(".jpg") or f.endswith(".png"):
            fpath = os.path.join(input_path, f)
            dfl_img = dfl.dfl_load_img(fpath)
            p, y, _ = dfl.dfl_estimate_pitch_yaw_roll(dfl_img)
            fno = int(f.split(".")[0])
            img_list.append([fno, p, y])
    # for i in range(10000):
    #     img_list.append([i,
    #                      random.random() * 2 - 1,
    #                      random.random() * 2 - 1])
    src_img_list = []
    src_cur_list = []
    img_list = np.array(img_list, "float")
    cur_list = img_list
    src_r = width / 100 * 2.5
    redius = width / 100 * 2

    trans_pitch_to_x = cv.trans_fn(-1, 1, 0, width)
    trans_yaw_to_y = cv.trans_fn(-1, 1, 0, width)
    trans_x_to_pitch = cv.trans_fn(0, width, -1, 1)
    trans_y_to_yaw = cv.trans_fn(0, width, -1, 1)
    trans_r = cv.trans_fn(0, width, 0, 2)
    cur_pitch_yaw = img_list[:, 1:3]
    img = cv.cv_new((width, width))
    cur_w = 2
    cur_mid = (0, 0)

    def reload_src():
        nonlocal src_img_list
        nonlocal src_cur_list
        src_img_list = []
        if src_path:
            for f in io.progress_bar_generator(os.listdir(src_path), "Loading"):
                if f.endswith(".jpg") or f.endswith(".png"):
                    fpath = os.path.join(src_path, f)
                    dfl_img = dfl.dfl_load_img(fpath)
                    p, y, _ = dfl.dfl_estimate_pitch_yaw_roll(dfl_img)
                    src_img_list.append([fno, p, y])
                    src_img_list.append([fno, p, -y])
        src_img_list = np.array(src_img_list, "float")
        src_cur_list = src_img_list

    def repaint():
        nonlocal trans_pitch_to_x
        nonlocal trans_yaw_to_y
        nonlocal trans_x_to_pitch
        nonlocal trans_y_to_yaw
        nonlocal trans_r
        nonlocal src_cur_list
        nonlocal cur_list
        nonlocal cur_pitch_yaw
        nonlocal img
        nonlocal src_path
        mid = cur_mid
        w = cur_w
        sx = mid[0] - w / 2
        sy = mid[1] - w / 2
        ex = mid[0] + w / 2
        ey = mid[1] + w / 2
        idxs = (img_list[:, 1] >= sx) & \
               (img_list[:, 2] >= sy) & \
               (img_list[:, 1] <= ex) & \
               (img_list[:, 2] <= ey)
        cur_list = img_list[idxs]
        cur_pitch_yaw = cur_list[:, 1:3]
        if len(src_img_list) == 0:
            src_cur_list = []
        elif src_path:
            idxs = (src_img_list[:, 1] >= sx) & \
                   (src_img_list[:, 2] >= sy) & \
                   (src_img_list[:, 1] <= ex) & \
                   (src_img_list[:, 2] <= ey)
            src_cur_list = src_img_list[idxs]

        trans_pitch_to_x = cv.trans_fn(sx, ex, 0, width)
        trans_yaw_to_y = cv.trans_fn(sy, ey, 0, width)
        trans_x_to_pitch = cv.trans_fn(0, width, sx, ex)
        trans_y_to_yaw = cv.trans_fn(0, width, sy, ey)
        trans_r = cv.trans_fn(0, width, 0, w)
        img = cv.cv_new((width, width))

        min_fno = int(cur_list[0][0])
        max_fno = int(cur_list[-1][0])
        trans_color = cv.trans_fn(min_fno, max_fno, 0, 1)
        for _, p, y in src_cur_list:
            cv.cv_point(img, (trans_pitch_to_x(p), trans_yaw_to_y(y)), (192, 192, 192), src_r * 2 / cur_w)
        for f, p, y in cur_list:
            fno = int(f)
            h = trans_color(fno)
            s = 1
            v = 1
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            cv.cv_point(img, (trans_pitch_to_x(p), trans_yaw_to_y(y)), (b * 255, g * 255, r * 255), 2)
        cv2.imshow("select", img)

    def mouse_callback(event, x, y, flags, param):
        nonlocal cur_mid
        nonlocal cur_w
        x = trans_x_to_pitch(x)
        y = trans_y_to_yaw(y)
        if event == cv2.EVENT_LBUTTONDOWN:
            tr = trans_r(redius)
            point = np.array([[x, y]] * len(cur_pitch_yaw), "float")
            dist = np.linalg.norm(cur_pitch_yaw - point, axis=1)
            idxs = dist <= tr
            for f, _, _ in cur_list[idxs]:
                print(f)
                pass
            print("-----------------------------------------")
        elif event == cv2.EVENT_RBUTTONDOWN:
            cur_mid = (x, y)
            cur_w = cur_w * ratio
            repaint()
        elif event == cv2.EVENT_MBUTTONDOWN:
            cur_w = cur_w / ratio
            if cur_w >= 2:
                cur_w = 2
                cur_mid = (0, 0)
            repaint()

    reload_src()
    cv2.namedWindow("select")
    cv2.setMouseCallback("select", mouse_callback)
    while True:
        repaint()
        key = cv2.waitKey()
        if key == 13 or key == -1:
            break
        elif key == 114:
            reload_src()


def prepare(workspace, detector="s3fd", manual_fix=True):
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
        tmp_aligned = os.path.join(tmp_dir, "aligned")
        tmp_video_dir = os.path.join(tmp_dir, "video")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
            os.mkdir(tmp_video_dir)
        video = os.path.join(workspace, f)
        # 提取帧
        VideoEd.extract_video(video, tmp_dir, "png", 0)
        # 提取人脸
        if detector == "manual":
            print('\a')
        Extractor.main(tmp_dir, tmp_aligned, detector=detector, manual_fix=manual_fix)
        # fanseg
        Extractor.extract_fanseg(tmp_aligned)
        if detector != "manual":
            #     # 两组人脸匹配
            #     skip_by_pitch(os.path.join(workspace, "data_src", "aligned"), os.path.join(tmp_dir, "aligned"))
            # 排序
            dfl.dfl_sort_by_hist(tmp_aligned)
        # 保存video
        shutil.copy(video, tmp_video_dir)
        # 重命名
        fname = f.replace(ext, "")
        dst_dir = os.path.join(workspace, "data_dst_%s_%s" % (get_time_str(), fname))
        shutil.move(tmp_dir, dst_dir)
        # 移动video
        data_trash = os.path.join(workspace, "data_trash")
        if not os.path.exists(data_trash):
            os.mkdir(data_trash)
        shutil.move(video, data_trash)
    print("\a")


def train(workspace):
    import os
    model_dir = os.path.join(workspace, "model")
    for f in os.listdir(workspace):
        if not os.path.isdir(os.path.join(workspace, f)) or not f.startswith("data_dst_"):
            continue
        io.log_info(f)
        data_dst = os.path.join(workspace, f)
        data_src_aligned = os.path.join(data_dst, "src")
        if not os.path.exists(data_src_aligned):
            data_src_aligned = os.path.join(workspace, "data_src", "aligned")
        data_dst_aligned = os.path.join(data_dst, "aligned")
        # 训练
        dfl.dfl_train(data_src_aligned, data_dst_aligned, model_dir)
        return


def train_dst(workspace):
    import os
    model_dir = os.path.join(workspace, "model")
    data_src_aligned = os.path.join(workspace, "data_src", "aligned")
    data_dst_aligned = os.path.join(workspace, "data_dst", "aligned")
    # 训练
    dfl.dfl_train(data_src_aligned, data_dst_aligned, model_dir)


def convert(workspace, skip=True):
    import os
    for f in os.listdir(workspace):
        if not os.path.isdir(os.path.join(workspace, f)) or not f.startswith("data_dst_"):
            continue
        io.log_info(f)
        model_dir = os.path.join(workspace, "model")
        self_model_dir = os.path.join(workspace, f, "model")
        if os.path.exists(self_model_dir):
            io.log_info("Use Self Model")
            model_dir = self_model_dir
        data_dst = os.path.join(workspace, f)
        data_dst_merged = os.path.join(data_dst, "merged")
        data_dst_aligned = os.path.join(data_dst, "aligned")
        data_dst_video = os.path.join(data_dst, "video")
        refer_path = None
        for v in os.listdir(data_dst_video):
            if v.split(".")[-1] in ["mp4", "avi", "wmv", "mkv"]:
                refer_path = os.path.join(data_dst_video, v)
                break
        if not refer_path:
            io.log_err("No Refer File In " + data_dst_video)
            return
        # 恢复排序
        need_recover = True
        for img in os.listdir(data_dst_aligned):
            if img.endswith("_0.jpg") or img.endswith("_0.png"):
                need_recover = False
            break
        if need_recover:
            recover_filename(data_dst_aligned)
        # 如果data_dst里没有脸则extract
        has_img = False
        for img in os.listdir(data_dst):
            if img.endswith(".jpg") or img.endswith(".png"):
                has_img = True
                break
        if not has_img:
            dfl.dfl_extract_video(refer_path, data_dst)
        # 转换
        dfl.dfl_convert(data_dst, data_dst_merged, data_dst_aligned, model_dir)
        # ConverterMasked.enable_predef = enable_predef
        # 去掉没有脸的
        if skip:
            skip_no_face(data_dst)
        # 转mp4
        refer_name = ".".join(os.path.basename(refer_path).split(".")[:-1])
        result_path = os.path.join(workspace, "result_%s_%s.mp4" % (get_time_str(), refer_name))
        dfl.dfl_video_from_sequence(data_dst_merged, result_path, refer_path)
        # 移动到trash
        trash_dir = os.path.join(workspace, "data_trash")
        import shutil
        shutil.move(data_dst, trash_dir)


def convert_dst(workspace):
    import os
    model_dir = os.path.join(workspace, "model")
    data_dst = os.path.join(workspace, "data_dst")
    data_dst_merged = os.path.join(data_dst, "merged")
    data_dst_aligned = os.path.join(data_dst, "aligned")
    # 转换
    dfl.dfl_convert(data_dst, data_dst_merged, data_dst_aligned, model_dir)


def mp4(workspace, skip=False):
    import os
    for f in os.listdir(workspace):
        if not os.path.isdir(os.path.join(workspace, f)) or not f.startswith("data_dst_"):
            continue
        io.log_info(f)
        data_dst = os.path.join(workspace, f)
        data_dst_merged = os.path.join(data_dst, "merged")
        data_dst_aligned = os.path.join(data_dst, "aligned")
        data_dst_video = os.path.join(data_dst, "video")
        refer_path = None
        for v in os.listdir(data_dst_video):
            if v.split(".")[-1] in ["mp4", "avi", "wmv", "mkv"]:
                refer_path = os.path.join(data_dst_video, v)
                break
        if not refer_path:
            io.log_err("No Refer File In " + data_dst_video)
            return
        # 恢复排序
        need_recover = True
        for img in os.listdir(data_dst_aligned):
            if img.endswith("_0.jpg") or img.endswith("_0.png"):
                need_recover = False
        if need_recover:
            recover_filename(data_dst_aligned)
        # 如果data_dst里没有脸则extract
        has_img = False
        for img in os.listdir(data_dst):
            if img.endswith(".jpg") or img.endswith(".png"):
                has_img = True
                break
        if not has_img:
            dfl.dfl_extract_video(refer_path, data_dst)
        # 去掉没有脸的
        if skip:
            skip_no_face(data_dst)
        # 转mp4
        refer_name = ".".join(os.path.basename(refer_path).split(".")[:-1])
        result_path = os.path.join(workspace, "result_%s_%s.mp4" % (get_time_str(), refer_name))
        dfl.dfl_video_from_sequence(data_dst_merged, result_path, refer_path)
        # 移动到trash
        trash_dir = os.path.join(workspace, "data_trash")
        import shutil
        shutil.move(data_dst, trash_dir)


def step(workspace):
    import shutil
    for f in os.listdir(workspace):
        if os.path.isdir(os.path.join(workspace, f)) and f.startswith("data_dst_"):
            model = os.path.join(workspace, "model")
            model_dst = os.path.join(workspace, f, "model")
            if not os.path.exists(model_dst):
                io.log_info("Move Model Files To %s" % f)
                os.mkdir(model_dst)
                for m in os.listdir(model):
                    mf = os.path.join(model, m)
                    if os.path.isfile(mf):
                        shutil.copy(os.path.join(model, m), model_dst)
            src = os.path.join(workspace, f)
            dst = os.path.join(workspace, "data_trash")
            io.log_info("Move %s To %s" % (src, dst))
            shutil.move(src, dst)
            return


def auto(workspace):
    import subprocess
    for f in os.listdir(workspace):
        if os.path.isdir(os.path.join(get_root_path(), "workspace", f)) and f.startswith("data_dst_"):
            train_bat = os.path.join(get_root_path(), "auto_train.bat")
            convert_bat = os.path.join(get_root_path(), "auto_convert.bat")
            step_bat = os.path.join(get_root_path(), "auto_step.bat")
            subprocess.call([train_bat])
            subprocess.call([convert_bat])
            subprocess.call([step_bat])
            io.log_info("Finish " + f)


def select(exists_path, pool_path, div=200):
    # 先计算output_path的已有图像
    import cv
    import dfl
    import random
    width = 800
    trans = cv.trans_fn(-1, 1, 0, width)
    img = cv.cv_new((width, width))
    for f in io.progress_bar_generator(os.listdir(exists_path), "Existing Imgs"):
        if f.endswith(".png") or f.endswith("jpg"):
            img_path = os.path.join(exists_path, f)
            dfl_img = dfl.dfl_load_img(img_path)
            pitch, yaw, _ = dfl.dfl_estimate_pitch_yaw_roll(dfl_img)
            pitch = trans(pitch)
            yaw = trans(yaw)
            cv.cv_circle(img, (pitch, yaw), (128, 128, 128), width / div, -1)
    time_str = get_time_str()
    import shutil
    pool_files = list(os.listdir(pool_path))
    # random.shuffle(pool_files)
    count = 0
    for f in io.progress_bar_generator(pool_files, os.path.basename(pool_path)):
        if f.endswith(".png") or f.endswith(".jpg"):
            img_path = os.path.join(pool_path, f)
            dfl_img = dfl.dfl_load_img(img_path)
            pitch, yaw, _ = dfl.dfl_estimate_pitch_yaw_roll(dfl_img)
            pitch = trans(pitch)
            yaw = trans(yaw)
            if sum(img[yaw][pitch]) == 255 * 3:
                dst = os.path.join(exists_path, "%s_%s" % (time_str, f))
                shutil.copy(img_path, dst)
                count += 1
                cv.cv_circle(img, (pitch, yaw), (0xcc, 0x66, 0x33), width / div, -1)
    cv.cv_save(img, os.path.join(exists_path, "_select.bmp"))
    io.log_info("Copy %d, Total %d" % (count, len(pool_files)))


def sync_trash(trash_path, pool_path):
    import shutil
    count = 0
    for f in io.progress_bar_generator(os.listdir(trash_path), "Trash Files"):
        if f.endswith(".jpg") or f.endswith(".png"):
            img_name = f.split("_")[-1]
            img_path = os.path.join(pool_path, img_name)
            dst_path = os.path.join(trash_path, "_origin")
            if os.path.exists(img_path):
                shutil.move(img_path, dst_path)
                count += 1
    io.log_info("Trash %d" % count)


def get_first_dst(workspace):
    for f in os.listdir(workspace):
        if f.startswith("data_dst_"):
            return os.path.join(workspace, f)


def auto_skip_by_pitch():
    workspace = os.path.join(get_root_path(), "workspace")
    for f in os.listdir(workspace):
        if f.startswith("data_dst_"):
            dst = os.path.join(workspace, f, "aligned")
            src = os.path.join(workspace, "data_src/aligned")
            skip_by_pitch(src, dst)
            break


def auto_extract_to_img():
    workspace = os.path.join(get_root_path(), "workspace")
    data_dst = None
    for f in os.listdir(workspace):
        if f.startswith("data_dst_"):
            data_dst = f
            break
    io.log_info(data_dst)
    video_name = None
    if data_dst is not None:
        name = "_".join(data_dst.split("_")[8:])
        print(name)
        for f in os.listdir(os.path.join(workspace, "data_trash")):
            if f.startswith(name):
                video_name = f
                break
    io.log_info(video_name)
    if video_name is not None:
        video_path = os.path.join(workspace, "data_trash", video_name)
        data_dst_path = os.path.join(workspace, data_dst)
        io.log_info(video_path)
        io.log_info(data_dst_path)
        for f in io.progress_bar_generator(os.listdir(data_dst_path), "Remove"):
            if f.endswith(".jpg") or f.endswith(".png"):
                os.remove(os.path.join(data_dst_path, f))
        dfl.dfl_extract_video(video_path, data_dst_path)


def fanseg(align_dir):
    from mainscripts import Extractor
    Extractor.extract_fanseg(align_dir)


def auto_fanseg():
    from mainscripts import Extractor
    workspace = os.path.join(get_root_path(), "workspace")
    for f in os.listdir(workspace):
        if f.startswith("data_dst_"):
            dst = os.path.join(workspace, f, "aligned")
            Extractor.extract_fanseg(dst)
            break


def merge_dst_aligned():
    import shutil
    workspace = os.path.join(get_root_path(), "workspace")
    counter = 0
    target_dst = os.path.join(workspace, "data_dst")
    if os.path.exists(target_dst):
        shutil.rmtree(target_dst)
    target_dst_aligned = os.path.join(target_dst, "aligned")
    os.makedirs(target_dst_aligned)
    for f in os.listdir(workspace):
        dst_path = os.path.join(workspace, f)
        if os.path.isdir(dst_path) and f.startswith("data_dst_"):
            counter += 1
            dst_aligned = os.path.join(dst_path, "aligned")
            for img in io.progress_bar_generator(os.listdir(dst_aligned), "Process"):
                if img.endswith(".png") or img.endswith(".jpg"):
                    img_path = os.path.join(dst_aligned, img)
                    base_name = os.path.basename(img_path)
                    dst_img_path = os.path.join(target_dst_aligned, "%d_%s" % (counter, base_name))
                    shutil.copy(img_path, dst_img_path)


def change_workspace():
    wss = []
    for f in os.listdir(get_root_path()):
        fpath = os.path.join(get_root_path(), f)
        if os.path.isfile(fpath) and f.startswith("@workspace"):
            os.remove(fpath)
        elif os.path.isdir(fpath) and f.startswith("workspace"):
            wss.append(f)
    inputs = "1234567890"[0:len(wss)]
    for i in range(0, len(wss)):
        io.log_info("[ %s ] %s" % (inputs[i], wss[i]))
    no = io.input_str("Select Workspace:", 0)[0]
    idx = inputs.find(no)
    if idx < 0:
        raise Exception("Invalid Idx " + no)
    ws = wss[idx]
    io.log_info("Select " + ws)
    f = open(os.path.join(get_root_path(), "@" + ws), 'w')
    f.write(ws)
    f.close()


def get_workspace():
    for f in os.listdir(get_root_path()):
        fpath = os.path.join(get_root_path(), f)
        if os.path.isfile(fpath) and f.startswith("@workspace"):
            return os.path.join(get_root_path(), f[1:])
    raise Exception("No @Workspace File")


def main():
    import sys

    arg = sys.argv[-1]
    if arg == '--skip-no-face':
        skip_no_face(os.path.join(get_root_path(), "workspace", "data_dst"))
    elif arg == '--change-workspace':
        change_workspace()
    elif arg == '--extract':
        extract()
    elif arg == '--extract-dst-image':
        extract_dst_image(get_workspace())
        train_dst(get_workspace())
        convert_dst(get_workspace())
    elif arg == '--prepare':
        prepare(get_workspace())
    elif arg == '--prepare-train':
        prepare(get_workspace())
        train(get_workspace())
        convert(get_workspace(), False)
    elif arg == '--prepare-nofix-train':
        prepare(get_workspace(), manual_fix=False)
        train(get_workspace())
        convert(get_workspace(), False)
    elif arg == '--prepare-manual-train':
        prepare(get_workspace(), "manual")
        train(get_workspace())
        convert(get_workspace(), False)
    elif arg == '--train':
        train(get_workspace())
        convert(get_workspace(), False)
    elif arg == '--train-dst':
        train_dst(get_workspace())
        convert_dst(get_workspace())
    # elif arg == '--convert':
    #     convert(get_workspace())
    #     # mp4(get_workspace())
    # elif arg == '--convert-no-skip':
    #     convert(get_workspace(), False)
    #     # mp4(get_workspace())
    elif arg == '--convert-skip-manual':
        convert(get_workspace(), skip=True)
    elif arg == '--convert-no-skip-manual':
        convert(get_workspace(), skip=False)
    elif arg == '--convert-dst':
        convert_dst(get_workspace())
    elif arg == '--mp4':
        mp4(get_workspace())
    elif arg == '--mp4-skip':
        mp4(get_workspace(), True)
    elif arg == '--step':
        step(get_workspace())
    elif arg == '--auto':
        auto(get_workspace())
    elif arg == '--merge-dst-aligned':
        merge_dst_aligned()
    elif arg == '--test':
        # prepare(os.path.join(get_root_path(), "workspace"), "manual")
        # match_by_pitch(os.path.join(get_root_path(), "workspace/data_src"),
        #                get_first_dst(os.path.join(get_root_path(), "workspace")))
        # sync_trash(os.path.join(get_root_path(), "extract_workspace/aligned_ym_4k_trash"),
        #            os.path.join(get_root_path(), "extract_workspace/aligned_ym_4k_all"))

        # select(os.path.join(get_root_path(), "extract_workspace/aligned_ym_4k_select"),
        #        os.path.join(get_root_path(), "extract_workspace/aligned_ym_4k_all"))

        # skip_by_pitch(os.path.join(get_root_path(), "workspace/data_src/aligned"),
        #               os.path.join(get_root_path(), "workspace/data_dst/aligned"))
        # get_pitch_yaw_roll(os.path.join(get_root_path(), "workspace_test", "data_src/aligned"))
        # split(os.path.join(get_root_path(), "extract_workspace/_/_chuang_ye_4k/all"),
        #       os.path.join(get_root_path(), "extract_workspace/_/_chuang_ye_4k/split"), 3000)
        # merge(os.path.join(get_root_path(), "extract_workspace/aligned_reba_all/fin"),
        #       os.path.join(get_root_path(), "extract_workspace/aligned_reba_all"),)
        # dfl.dfl_sort_by_hist(os.path.join(get_root_path(), "extract_workspace/_/_san_sheng_4k/all"))
        # get_pitch_yaw_roll(os.path.join(get_root_path(), "extract_workspace/aligned_ym_4k_all"))
        # get_pitch_yaw_roll(os.path.join(get_root_path(), "workspace/data_src/aligned"))
        manual_select(os.path.join(get_root_path(), "extract_workspace/aligned_ab_ex4"),
                      os.path.join(get_root_path(), "workspace/data_src/aligned"))
        # manual_select(os.path.join(get_root_path(), "workspace/data_src/aligned"),
        #               os.path.join(get_root_path(), "workspace/data_src/aligned"))
        # dfl.dfl_sort_by_hist(os.path.join(get_root_path(), "extract_workspace/aligned_ab_all"))
        # auto_skip_by_pitch()
        # auto_extract_to_img()
        # fanseg(os.path.join(get_root_path(), "extract_workspace", "aligned_ty"))
        # get_pitch_yaw_roll(os.path.join(get_root_path(), "extract_workspace", "aligned_fj_all"))
        # dfl.dfl_sort_by_hist(os.path.join(get_root_path(), "extract_workspace", "aligned_ty"))
        # split(os.path.join(get_root_path(), "extract_workspace/aligned_lyf"),
        #       os.path.join(get_root_path(), "extract_workspace/aligned_lyf")
        #       )
        # merge_dst_aligned()
        pass


if __name__ == '__main__':
    main()
