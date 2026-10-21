"""环境自检工具。"""
import torch


def assert_npu_ready():
    """检查 torch_npu 可用且设备可初始化，不可用则抛出带定位信息的异常。"""
    try:
        import torch_npu  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "torch_npu 未安装: 请使用 CANN 官方镜像(如 quay.io/ascend/vllm-ascend), "
            "不要用 pip 安装 torch 覆盖") from e
    n = torch.npu.device_count()
    if n < 1:
        raise RuntimeError(
            "NPU 初始化失败(device_count=0)。常见原因: "
            "1) 驱动按容器命名空间独占设备, 本容器未认领到卡 "
            "(dmesg 查 'Conflict open udevid'); "
            "2) ASCEND_VISIBLE_DEVICES 指向了已被占用的卡 "
            "(npu-smi info 选 Hugepages 0/0 且无进程的卡); "
            "3) 容器内未 source set_env.sh")
    return n


def print_env():
    """打印 torch/torch_npu/CANN 版本与设备信息。"""
    import torch_npu
    info = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device_count": torch.npu.device_count(),
        "device_name": torch.npu.get_device_name(0),
    }
    for k, v in info.items():
        print(f"  {k}: {v}")
    return info
