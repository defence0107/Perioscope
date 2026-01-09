import torch
import psutil  # 用于获取系统内存信息

def print_memory_release():
    # 记录初始内存使用
    initial_memory = psutil.virtual_memory().used
    print(f"初始内存使用: {initial_memory} 字节")


    # 释放 GPU 内存
    torch.cuda.empty_cache()

    # 记录释放后的内存使用
    after_release_memory = psutil.virtual_memory().used
    print(f"释放后的内存使用: {after_release_memory} 字节")
    print(f"释放的内存量: {initial_memory - after_release_memory} 字节")

print_memory_release()