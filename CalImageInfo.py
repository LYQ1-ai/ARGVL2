import cv2
import numpy as np
from pathlib import Path
import seaborn as sns

from matplotlib import pyplot as plt
from scipy.stats import skew, kurtosis
from tqdm import tqdm


def calculate_image_information(image_path, num_bands=3, target_size=(224, 224)):
    """计算压缩后图片的频域信息量（熵）"""
    try:
        # 1. 读取图像并压缩
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"警告：无法读取图片 {image_path.name}，已跳过")
            return None

        # 压缩至 3x224x224（RGB通道，HxW=224x224）
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)  # 抗锯齿压缩

        # 转换为灰度图（若需保留彩色通道需修改FFT计算逻辑）
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        img_normalized = img_gray.astype(np.float32) / 255.0

        # 2. FFT变换与频带划分（后续逻辑不变）
        fft = np.fft.fft2(img_normalized)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)

        h, w = img_normalized.shape
        cy, cx = h // 2, w // 2
        y_coords, x_coords = np.indices((h, w))
        distances = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)
        max_r = np.sqrt(cy ** 2 + cx ** 2)

        band_edges = np.linspace(0, max_r, num_bands + 1)
        bands = np.digitize(distances, band_edges) - 1
        bands = np.clip(bands, 0, num_bands - 1)

        # 计算频带能量熵
        energy = magnitude  **  2 + 1e-10
        total_energy = np.sum(energy)
        energy_bands = [np.sum(energy[bands == i]) for i in range(num_bands)]
        p = np.array(energy_bands) / total_energy
        entropy = -np.sum(p * np.log2(p))
        return entropy
    except Exception as e:
        print(f"处理图片 {image_path.name} 时发生错误: {str(e)}")
        return None


def calculate_average_information(image_dir, num_bands=3):
    """统计目录下所有图片的平均信息量"""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"目录 {image_dir} 不存在")

    # 支持常见图片格式
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend(image_dir.glob(ext))

    if not image_paths:
        print("警告：目录中未找到图片文件")
        return 0.0

    # 计算每张图片的信息量
    entropies = []
    for path in image_paths:
        entropy = calculate_image_information(path, num_bands)
        if entropy is not None:
            entropies.append(entropy)
            print(f"图片 {path.name:20} 信息量: {entropy:.4f}")

    if not entropies:
        print("错误：未成功计算任何图片的信息量")
        return 0.0

    # 计算平均值
    avg_entropy = np.mean(entropies)
    print("\n统计结果:")
    print(f"- 图片数量: {len(entropies)}")
    print(f"- 平均信息量: {avg_entropy:.4f}")
    return avg_entropy


def analyze_information_distribution(image_dir, num_bands=3, save_plot=True,fig_file_name="information_distribution.png"):
    # 获取所有图片信息量
    image_dir = Path(image_dir)
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend(image_dir.glob(ext))

    entropies = []
    for path in tqdm(image_paths):
        entropy = calculate_image_information(path, num_bands)
        if entropy is not None:
            entropies.append(entropy)

    if not entropies:
        print("无有效数据可分析")
        return

    # 转换为NumPy数组
    entropies = np.array(entropies)

    # 统计特征
    stats = {
        "count": len(entropies),
        "mean": np.mean(entropies),
        "std": np.std(entropies),
        "min": np.min(entropies),
        "25%": np.percentile(entropies, 25),
        "50%": np.median(entropies),
        "75%": np.percentile(entropies, 75),
        "max": np.max(entropies),
        "skewness": skew(entropies),
        "kurtosis": kurtosis(entropies)
    }

    # 打印统计结果
    print("\n信息量分布统计:")
    for key, value in stats.items():
        print(f"- {key:8}: {value:.4f}")

    # 可视化
    plt.figure(figsize=(15, 5))

    # 子图1：直方图 + KDE
    plt.subplot(1, 3, 1)
    sns.histplot(entropies, kde=True, color="skyblue", bins=20, edgecolor="black")
    plt.title("Histogram with KDE")
    plt.xlabel("Information Entropy")
    plt.ylabel("Frequency")

    # 子图2：箱线图
    plt.subplot(1, 3, 2)
    sns.boxplot(y=entropies, color="lightgreen")
    plt.title("Boxplot")
    plt.ylabel("Information Entropy")

    # 子图3：累积分布函数（CDF）
    plt.subplot(1, 3, 3)
    sorted_entropies = np.sort(entropies)
    cdf = np.arange(1, len(sorted_entropies) + 1) / len(sorted_entropies)
    plt.plot(sorted_entropies, cdf, marker='.', linestyle='none', color="orange")
    plt.title("Cumulative Distribution Function (CDF)")
    plt.xlabel("Information Entropy")
    plt.ylabel("CDF")

    plt.tight_layout()

    if save_plot:
        plt.savefig(fig_file_name, dpi=300, bbox_inches="tight")
    plt.show()


# 示例调用
if __name__ == "__main__":
    image_directory = "C:\\Users\\lyq\DataSet\\FakeNews\\gossipcop\\images"
    analyze_information_distribution(image_directory, num_bands=3, save_plot=True,fig_file_name='gossipcop_information_distribution.png')