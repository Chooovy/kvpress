import time
import argparse
import sys
import os
import subprocess
import multiprocessing

def check_other_processes(gpu_id):
    """
    Check if there are other processes running on the GPU.
    Returns True if other processes are detected.
    """
    try:
        # Query compute processes on the specific GPU
        # We use nvidia-smi to get PIDs of processes using the GPU
        # --format=csv,noheader,nounits ensures we get clean output
        cmd = ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits', '-i', str(gpu_id)]
        output = subprocess.check_output(cmd).decode('utf-8').strip()

        if not output:
            return False
            
        pids = [int(x) for x in output.split('\n') if x.strip()]
        my_pid = os.getpid()
        
        # Filter out my_pid
        other_pids = [p for p in pids if p != my_pid]
        
        if not other_pids:
            return False
            
        # If we are here, we have PIDs that are not my_pid.
        # If the total number of processes on GPU is 1, and we are running, 
        # that 1 process is likely us (PID mismatch due to container).
        if len(pids) == 1:
            return False
            
        return True
                
    except Exception:
        # If nvidia-smi fails or is not found, we can't check.
        # We assume no other processes to avoid interrupting constantly,
        # but in a real strict environment you might want to yield.
        pass
    return False

def gpu_worker(physical_gpu_id, memory_gb, utilization, interval):
    # Use environment variable to isolate the process to a specific physical GPU.
    # This prevents the process from accidentally initializing/touching other GPUs (like GPU 0).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    
    try:
        import torch
    except ImportError:
        print("Error: torch is required for GPU occupation. Please install it via 'pip install torch'.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print(f"[Physical GPU {physical_gpu_id}] Error: CUDA is not available.")
        return

    # After setting CUDA_VISIBLE_DEVICES, the selected GPU is seen as cuda:0.
    device = torch.device("cuda:0")
    print(f"[Physical GPU {physical_gpu_id}] Process {os.getpid()} started.")
    print(f"[Physical GPU {physical_gpu_id}] Target: {memory_gb}GB memory, {utilization} utilization. Check interval: {interval}s")

    memory_tensor = None
    
    # Matrix size for computation
    n = 2048 
    try:
        a = torch.randn(n, n, device=device)
        b = torch.randn(n, n, device=device)
    except Exception as e:
        print(f"[Physical GPU {physical_gpu_id}] Error initializing tensors: {e}")
        return

    try:
        while True:
            # 1. Check for other processes on the PHYSICAL GPU
            if check_other_processes(physical_gpu_id):
                if memory_tensor is not None:
                    print(f"[Physical GPU {physical_gpu_id}] Detected other activity. Releasing resources...")
                    del memory_tensor
                    memory_tensor = None
                    torch.cuda.empty_cache()
                
                # Sleep and wait for resources to free up
                time.sleep(interval)
                continue

            # 2. If free, occupy resources
            if memory_tensor is None:
                print(f"[Physical GPU {physical_gpu_id}] GPU is free. Allocating {memory_gb}GB...")
                try:
                    num_elements = int(memory_gb * (1024**3) / 4)
                    if num_elements > 0:
                        memory_tensor = torch.ones(num_elements, dtype=torch.float32, device=device)
                except RuntimeError as e:
                    print(f"[Physical GPU {physical_gpu_id}] Failed to allocate memory: {e}")
                    print(f"[Physical GPU {physical_gpu_id}] Continuing with computation only...")

            # 3. Compute burst
            # Run computation for 'interval' seconds, then check again
            start_burst = time.time()
            while time.time() - start_burst < interval:
                start_op = time.time()
                
                # Perform computation
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
                
                elapsed = time.time() - start_op
                
                # Control utilization
                if utilization < 1.0 and utilization > 0.0:
                    sleep_time = elapsed * (1.0 / utilization - 1.0)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                elif utilization <= 0.0:
                    time.sleep(1)
                
                # Break inner loop if interval exceeded
                if time.time() - start_burst >= interval:
                    break
                
    except KeyboardInterrupt:
        print(f"[Physical GPU {physical_gpu_id}] Stopped.")
    except Exception as e:
        print(f"[Physical GPU {physical_gpu_id}] Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Smart Multi-GPU Occupy Script")
    parser.add_argument("--memory", type=float, default=0.5, help="Memory to occupy per GPU in GB (default: 2.0)")
    parser.add_argument("--util", type=float, default=0.95, help="Target utilization 0.0-1.0 (default: 0.95)")
    parser.add_argument("--interval", type=int, default=10, help="Check interval in seconds (default: 10)")
    parser.add_argument("--delay", type=float, default=0, help="Delay in minutes before starting detection and occupation (default: 0)")
    
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("Error: torch is required. Please install it via 'pip install torch'.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("Error: CUDA is not available.")
        sys.exit(1)

    # Get the mapping of local indices to physical indices based on CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        # If CUDA_VISIBLE_DEVICES is set, physical IDs are the ones provided in the list
        physical_ids = [x.strip() for x in cuda_visible.split(',') if x.strip()]
    else:
        # Otherwise, physical IDs are just 0, 1, 2...
        num_gpus = torch.cuda.device_count()
        physical_ids = [str(i) for i in range(num_gpus)]

    print(f"Detected {len(physical_ids)} GPUs: {', '.join(physical_ids)}")

    if args.delay > 0:
        print(f"Delaying {args.delay} min before starting...")
        time.sleep(args.delay * 60)

    processes = []
    for physical_id in physical_ids:
        p = multiprocessing.Process(target=gpu_worker, args=(physical_id, args.memory, args.util, args.interval))
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nTerminating all processes...")
        for p in processes:
            p.terminate()
            p.join()

if __name__ == "__main__":
    # Use spawn to avoid CUDA initialization issues in forked processes
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
