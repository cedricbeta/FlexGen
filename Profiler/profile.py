import GPUtil
import psutil
import argparse
from diskspeed import DiskSpeed


def report_system_status(drive_name="/dev/sda2"):
    # RAM Utilization
    ram = psutil.virtual_memory()
    print("===== RAM Utilization =====")
    print(f"RAM memory % used: {ram.percent}")
    print(f"RAM Total (GB): {ram.total / (1024**3):.2f}")
    print(f"RAM Used (GB): {ram.used / (1024**3):.2f}")
    print(f"RAM Free (GB): {ram.free / (1024**3):.2f}")
    print("\n")
    
    # GPU Utilization
    GPUs = GPUtil.getGPUs()
    print("===== GPU Utilization =====")
    GPUtil.showUtilization()

    # Available GPUs based on load and memory
    GPUavailability = GPUtil.getAvailability(GPUs, maxLoad=0.5, maxMemory=0.5)
    print(f"Available GPUs (IDs): {GPUavailability}")

    # Display memory for each GPU
    for gpu in GPUs:
        print(f"GPU {gpu.id} {gpu.name}: Memory Free = {gpu.memoryFree}MB")
    print("\n")
    
    # CPU Utilization
    cpu_percent = psutil.cpu_percent()
    print("===== CPU Utilization =====")
    print(f"CPU % used: {cpu_percent}")
    print("\n")
    
    # Disk Utilization
    disk = psutil.disk_usage('/')
    print("===== Disk Utilization =====")
    print(f"Disk Total (GB): {disk.total / (1024**3):.2f}")
    print(f"Disk Used (GB): {disk.used / (1024**3):.2f}")
    print(f"Disk Free (GB): {disk.free / (1024**3):.2f}")
    print(f"Disk % Used: {disk.percent}")
    
    # Disk Bandwidth
    disk_speed = DiskSpeed(drive_name)
    print("\n===== Disk Speed (from DiskSpeed class) =====")
    print(f"Disk write speed for drive {drive_name} is {disk_speed.get_write_speed() } bits/s")
    print(f"Disk read speed for drive {drive_name} is {disk_speed.get_read_speed() } bits/s")
    # print(f"Read/Write speed for drive {drive_name} is: read speed = {disk_speed.get_read_write_speed()['read']} bits/s, write speed = {disk_speed.get_read_write_speed()['write']} bits/s")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="System status reporter")
    parser.add_argument("--drive", type=str, default="/dev/sda2", help="Drive name for speed testing.")
    args = parser.parse_args()
    report_system_status(args.drive)
