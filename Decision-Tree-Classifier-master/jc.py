import os
import pickle
import hashlib


def calculate_file_hash(file_path, algorithm='sha256'):
    """计算文件的哈希值以验证文件完整性"""
    if not os.path.isfile(file_path):
        return None

    hash_obj = hashlib.new(algorithm)
    try:
        with open(file_path, 'rb') as f:
            # 分块读取以处理大文件
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except Exception as e:
        print(f"计算哈希值时出错: {e}")
        return None


def check_model_integrity(model_path):
    """检查模型文件是否损坏"""
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 文件 '{model_path}' 不存在")
        return False

    # 检查文件大小是否合理（非零）
    file_size = os.path.getsize(model_path)
    if file_size == 0:
        print(f"错误: 文件 '{model_path}' 为空")
        return False

    # 尝试加载模型
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"文件 '{model_path}' 完好，可以正常加载。")
        print(f"模型类型: {type(model).__name__}")

        # 计算并打印文件哈希值（用于未来验证）
        file_hash = calculate_file_hash(model_path)
        if file_hash:
            print(f"文件 SHA-256 哈希值: {file_hash}")

        return True
    except pickle.UnpicklingError as e:
        print(f"错误: 文件 '{model_path}' 损坏，无法反序列化: {e}")
    except Exception as e:
        print(f"错误: 加载文件 '{model_path}' 时发生未知错误: {e}")

    return False


if __name__ == "__main__":
    # 指定要检查的模型文件路径
    model_file = "random_forest_model_final.pkl"

    # 检查模型完整性
    check_model_integrity(model_file)