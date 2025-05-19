import torch
import platform
import os
import subprocess

def get_cpu_info():
    cpu_info = {}
    if platform.system() == "Windows":
        # Windows detection
        import ctypes
        cpu_info["cores"] = os.cpu_count()
        # Use WMI to get clock speed
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
                freq_line = next(line for line in cpuinfo.split('\n') if 'cpu MHz' in line)
                freq_ghz = float(freq_line.split(':')[1].strip()) / 1000
                cpu_info["clock_ghz"] = freq_ghz / 1000
        except:
            cpu_info["clock_ghz"] = None
    
    elif platform.system() == "Linux":
        # Linux detection
        cpu_info["cores"] = os.cpu_count()
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "MHz" in line:
                        clock_mhz = float(line.split(":")[1].strip())
                        cpu_info["clock_ghz"] = clock_mhz / 1000
                        break
        except:
            cpu_info["clock_ghz"] = None
    
    elif platform.system() == "Darwin":
        # macOS detection
        cpu_info["cores"] = os.cpu_count()
        try:
            output = subprocess.check_output(["sysctl", "-n", "hw.cpufrequency_max"]).decode().strip()
            cpu_info["clock_ghz"] = float(output) / 1e9
        except:
            cpu_info["clock_ghz"] = None
    
    # Assume FLOPS per cycle based on architecture
    cpu_info["flops_per_cycle"] = 32  # Common for modern CPUs
    
    return cpu_info

def theoretical_tflops(clock_speed_ghz, cores, flops_per_cycle):
    flops = clock_speed_ghz * cores * flops_per_cycle * 1e9  # FLOPS
    tflops = flops / 1e12
    return round(tflops, 3)


def get_gpu_info():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        # Get GPU name
        gpu_name = props.name
        
        # Calculate CUDA cores based on GPU architecture
        sm_count = props.multi_processor_count
        # This is an approximation - actual count varies by architecture
        cuda_cores_per_sm = 128  # Varies by GPU generation
        cores = sm_count * cuda_cores_per_sm
        
        # Get clock speed in GHz
        clock_mhz = props.clock_rate / 1000
        clock_ghz = clock_mhz / 1000
        
        # Most NVIDIA GPUs can do 2 FMA operations per cycle
        flops_per_cycle = 2
        
        return {
            "name": gpu_name,
            "cores": cores,
            "clock_ghz": clock_ghz,
            "flops_per_cycle": flops_per_cycle
        }
    return None

def get_flops():
        
    # Get GPU info and calculate TFLOPS
    gpu_info = get_gpu_info()
    if gpu_info:        
        gpu_tflops = theoretical_tflops(
            gpu_info["clock_ghz"],
            gpu_info["cores"],
            gpu_info["flops_per_cycle"]
        )
        return gpu_tflops
    
    # Get CPU info
    cpu_info = get_cpu_info()

    # Calculate TFLOPS
    cpu_tflops = theoretical_tflops(
        cpu_info["clock_ghz"], 
        cpu_info["cores"], 
        cpu_info["flops_per_cycle"]
    )
    return cpu_tflops, cpu_info


if __name__ == '__main__':
    print(get_flops())