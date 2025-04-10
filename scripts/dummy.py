import pynvml
import time
import torch
import signal
import sys
from multiprocessing import Process, Event, Queue
import argparse
import logging

class GPUMonitor:
    def __init__(self, gpu_indices, check_interval=5, verbose=False):
        self.gpu_indices = gpu_indices
        self.check_interval = check_interval
        self.processes = []
        self.stop_event = Event()
        self.verbose = verbose
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def dummy_work(self, gpu_idx, pause_event):
        """GPU负载生成函数"""
        torch.cuda.set_device(gpu_idx)
        device = torch.device(f'cuda:{gpu_idx}')
        
        while not self.stop_event.is_set():
            while not pause_event.is_set() and not self.stop_event.is_set():
                # 动态调整计算参数
                size = 1000
                a = torch.randn(size, size, device=device)
                b = torch.randn(size, size, device=device)
                torch.mm(a, b)
                time.sleep(1e-4)  # 调整间隔控制利用率
            time.sleep(0.1)  # 暂停时降低检查频率

    def monitor(self, gpu_idx, control_queue):
        """GPU监控进程"""
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        pause_event = Event()
        dummy_proc = None

        try:
            # 启动负载生成进程
            pause_event.clear()
            dummy_proc = Process(target=self.dummy_work, args=(gpu_idx, pause_event))
            dummy_proc.start()

            low_util_count = 0
            while not self.stop_event.is_set():
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                # control_queue.put((gpu_idx, util))
                if util < 30:
                    low_util_count += 1
                else:
                    low_util_count = 0
                
                if pause_event.is_set():
                    if low_util_count > 2:
                        pause_event.clear()
                        self.logger.info(f"GPU {gpu_idx}: Resumed dummy work ({util=}%)")
                    else:
                        self.logger.info(f"GPU {gpu_idx} Current Utilization: {util}%")
                else: # pause not set
                    if util > 60:
                        pause_event.set()
                        self.logger.info(f"GPU {gpu_idx}: Paused dummy work ({util=}%)")
                    else:
                        self.logger.info(f"GPU {gpu_idx} Current Utilization: {util}%")

                # if low_util_count > 1:
                #     if not pause_event.is_set():
                #         pause_event.set()
                #         self.logger.info(f"GPU {gpu_idx}: Paused dummy work (Util: {util}%)")
                #     else:
                #         self.logger.info(f"GPU {gpu_idx} Current Utilization: {util}%")
                # else:
                #     if pause_event.is_set():
                #         pause_event.clear()
                #         self.logger.info(f"GPU {gpu_idx}: Resumed dummy work (Util: {util}%)")
                #     else:
                #         self.logger.info(f"GPU {gpu_idx} Current Utilization: {util}%")

                time.sleep(self.check_interval)
        finally:
            if dummy_proc and dummy_proc.is_alive():
                dummy_proc.terminate()
            pynvml.nvmlShutdown()

    def start(self):
        """启动监控系统"""
        control_queue = Queue()
        
        # 启动监控进程
        for idx in self.gpu_indices:
            p = Process(target=self.monitor, args=(idx, control_queue))
            p.start()
            self.processes.append(p)
            time.sleep(0.2)

        # 控制台输出线程
        try:
            # while not self.stop_event.is_set():
                # while not control_queue.empty():
                #     gpu_idx, util = control_queue.get()
                #     self.logger.info(f"GPU {gpu_idx} Current Utilization: {util}%")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """停止所有进程"""
        self.stop_event.set()
        time.sleep(1)
        for p in self.processes:
            if p.is_alive():
                p.terminate()
        for p in self.processes:
            p.join()
        print("\nSystem safely stopped.")

    def signal_handler(self, sig, frame):
        """信号处理"""
        self.stop()
        sys.exit(0)

def parse_args():
    parser = argparse.ArgumentParser(description="Run dummy GPU utilization when GPUs are idle")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7", help="Comma-separated list of GPU IDs to use")
    parser.add_argument("--check-interval", type=int, default=20, help="Interval in seconds to check GPU usage")
    parser.add_argument("--verbose", action="store_true", default=False, help="Verbose level")
    # parser.add_argument("--utilization-threshold", type=int, default=10, help="GPU utilization threshold below which is considered idle (%)")
    # parser.add_argument("--tensor-size", type=int, default=300, help="Size of tensor for memory consumption (NxN)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    gpu_indices = [int(idx) for idx in args.gpus.split(",")]
    
    monitor = GPUMonitor(gpu_indices, check_interval=args.check_interval, verbose=args.verbose)
    monitor.start()