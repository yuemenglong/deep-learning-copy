import numpy as np
import cv2


def cv_sharp(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    img = cv2.filter2D(img, -1, kernel=kernel)
    return img


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
    import numpy as np
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


def face_encodings(image):
    import face_recognition
    import numpy as np
    res = face_recognition.face_encodings(image)
    if len(res) == 1:
        return res[0]
    else:
        return np.ones(128)


def face_distance(known_enc, unknown_enc):
    import face_recognition
    if not isinstance(known_enc, list):
        known_enc = [known_enc]
    score = face_recognition.face_distance(known_enc, unknown_enc)
    return np.min(score)


def cpu_count():
    from multiprocessing import cpu_count as cs
    return cs()


def sort_by_face_similarity(known_dir, target_dir):
    import os
    import shutil
    known_imgs = [cv2.imread(os.path.join(known_dir, f)) for f in os.listdir(known_dir)]
    known_enc = [face_encodings(i) for i in known_imgs]
    files = os.listdir(target_dir)
    imgs = [cv2.imread(os.path.join(target_dir, f)) for f in files]
    jobs = list(zip(files, imgs))

    for job in jobs:
        f, img = job
        img = cv2.imread(os.path.join(target_dir, f))
        enc = face_encodings(img)
        score = face_distance(known_enc, enc)
        ext = os.path.splitext(f)[-1]
        name = "%08d" % (score * 1e7) + ext
        src = os.path.join(target_dir, f)
        dst = os.path.join(target_dir, name)
        shutil.move(src, dst)
        print(src, dst)


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
        min_pixel = 512
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

    with open(os.path.join(input_path, "_pitch_yaw_roll.csv"), "w") as f:
        for i in img_list:
            f.write("%s,%f,%f,%f\n" % (i[0], i[1], i[2],i[3]))

    return img_list


def main():
    import sys
    import os

    arg = sys.argv[-1]
    if arg == '--skip-no-face':
        skip_no_face(os.path.join(get_root_path(), "workspace", "data_dst"))
    elif arg == '--extract':
        extract()
    else:
        get_pitch_yaw_roll(os.path.join(get_root_path(), "extract_workspace", "aligned_ym_4k_01_08"))
        pass


if __name__ == '__main__':
    main()
