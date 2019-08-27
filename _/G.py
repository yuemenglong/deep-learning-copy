def pick_spec(input_path, output_path):
    import os
    import numpy as np
    from sklearn.cluster import DBSCAN
    import shutil
    import matplotlib.pyplot as plt
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    csv_file = os.path.join(input_path, "_pitch_yaw_roll.csv")
    if not os.path.exists(csv_file):
        raise Exception("No Csv File")
    pitch_yaw = []
    paths = []
    with open(csv_file) as f:
        for line in f.readlines():
            [file, pitch, yaw, _roll] = line.strip().split(",")
            paths.append(file)
            pitch_yaw.append([float(pitch), float(yaw)])
    pitch_yaw = np.array(pitch_yaw)
    estimator = DBSCAN(eps=0.1, min_samples=len(pitch_yaw) / 100)  # 构造聚类器
    estimator.fit(pitch_yaw)  # 聚类
    label_pred = estimator.labels_  # 获取聚类标签
    plt.scatter(pitch_yaw[:, 0], pitch_yaw[:, 1], c=label_pred)
    plt.show()
    spec_idx = np.where(label_pred == -1)
    print(len(paths))
    print(len(spec_idx[0]))
    for idx in spec_idx[0]:
        print(paths[idx])
        shutil.copy(paths[idx], output_path)


def get_root_path():
    return "D:/DeepFaceLabCUDA10.1AVX"


def main():
    import os
    pick_spec(os.path.join(get_root_path(), "extract_workspace", "aligned_ab_01_20"),
              os.path.join(get_root_path(), "extract_workspace", "aligned_ab_pick"))


if __name__ == '__main__':
    main()
