#!/usr/bin/env python3
"""
CUDA Environment Verification Script

This script comprehensively checks the CUDA development environment for MLFF Distiller,
including GPU capabilities, driver versions, PyTorch CUDA support, and profiling tools.

Usage:
    python scripts/check_cuda.py
    python scripts/check_cuda.py --verbose
    python scripts/check_cuda.py --export-json gpu_specs.json
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def check_nvidia_smi() -> Tuple[bool, Optional[Dict]]:
    """Check NVIDIA GPU using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,compute_cap",
             "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                idx, name, driver, memory, compute_cap = [x.strip() for x in line.split(',')]
                gpus.append({
                    'index': int(idx),
                    'name': name,
                    'driver_version': driver,
                    'memory_total_mb': int(memory.split()[0]),
                    'compute_capability': compute_cap
                })

        return True, {'gpus': gpus}
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return False, {'error': str(e)}


def check_cuda_toolkit() -> Tuple[bool, Optional[Dict]]:
    """Check CUDA toolkit installation."""
    info = {}

    # Check nvcc
    nvcc_paths = [
        '/usr/local/cuda/bin/nvcc',
        '/usr/local/cuda-12.6/bin/nvcc',
        '/usr/local/cuda-12.1/bin/nvcc',
    ]

    # Also search PATH
    try:
        result = subprocess.run(['which', 'nvcc'], capture_output=True, text=True)
        if result.returncode == 0:
            nvcc_paths.insert(0, result.stdout.strip())
    except Exception:
        pass

    nvcc_found = False
    for nvcc_path in nvcc_paths:
        if os.path.exists(nvcc_path):
            try:
                result = subprocess.run(
                    [nvcc_path, '--version'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                # Parse CUDA version from output
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        version = line.split('release')[1].split(',')[0].strip()
                        info['nvcc_version'] = version
                        info['nvcc_path'] = nvcc_path
                        nvcc_found = True
                        break
                if nvcc_found:
                    break
            except subprocess.CalledProcessError:
                continue

    if not nvcc_found:
        info['nvcc_version'] = 'Not found'
        info['nvcc_path'] = 'Not found'

    # Check CUDA_HOME
    cuda_home = os.environ.get('CUDA_HOME', 'Not set')
    info['cuda_home'] = cuda_home

    # Check for CUDA libraries
    cuda_lib_paths = [
        '/usr/local/cuda/lib64',
        '/usr/local/cuda-12.6/lib64',
        '/usr/lib/x86_64-linux-gnu',
    ]

    cuda_libs = ['libcudart.so', 'libcublas.so', 'libcudnn.so']
    found_libs = {}

    for lib in cuda_libs:
        found = False
        for path in cuda_lib_paths:
            lib_path = Path(path) / lib
            if lib_path.exists():
                found_libs[lib] = str(lib_path)
                found = True
                break
        if not found:
            found_libs[lib] = 'Not found'

    info['cuda_libraries'] = found_libs

    return nvcc_found, info


def check_pytorch() -> Tuple[bool, Optional[Dict]]:
    """Check PyTorch and CUDA availability."""
    try:
        import torch

        info = {
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A',
            'cudnn_enabled': torch.backends.cudnn.enabled if torch.cuda.is_available() else False,
        }

        if torch.cuda.is_available():
            info['device_count'] = torch.cuda.device_count()
            info['devices'] = []

            for i in range(torch.cuda.device_count()):
                device_info = {
                    'index': i,
                    'name': torch.cuda.get_device_name(i),
                    'compute_capability': '.'.join(map(str, torch.cuda.get_device_capability(i))),
                    'total_memory_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3),
                    'multi_processor_count': torch.cuda.get_device_properties(i).multi_processor_count,
                }
                info['devices'].append(device_info)

        return torch.cuda.is_available(), info

    except ImportError as e:
        return False, {'error': f'PyTorch not installed: {e}'}


def check_profiling_tools() -> Dict[str, bool]:
    """Check availability of profiling tools."""
    tools = {}

    # Check nsys (Nsight Systems)
    try:
        result = subprocess.run(['nsys', '--version'], capture_output=True, text=True)
        tools['nsys'] = result.returncode == 0
        if tools['nsys']:
            version_line = result.stdout.split('\n')[0]
            tools['nsys_version'] = version_line
    except FileNotFoundError:
        tools['nsys'] = False
        tools['nsys_version'] = 'Not installed'

    # Check ncu (Nsight Compute)
    try:
        result = subprocess.run(['ncu', '--version'], capture_output=True, text=True)
        tools['ncu'] = result.returncode == 0
        if tools['ncu']:
            version_line = result.stdout.split('\n')[0]
            tools['ncu_version'] = version_line
    except FileNotFoundError:
        tools['ncu'] = False
        tools['ncu_version'] = 'Not installed'

    # Check torch.profiler availability
    try:
        import torch.profiler
        tools['torch_profiler'] = True
    except ImportError:
        tools['torch_profiler'] = False

    return tools


def check_optional_packages() -> Dict[str, Dict]:
    """Check optional CUDA-related Python packages."""
    packages = {}

    # CuPy
    try:
        import cupy
        packages['cupy'] = {
            'installed': True,
            'version': cupy.__version__,
            'cuda_version': cupy.cuda.runtime.runtimeGetVersion()
        }
    except ImportError:
        packages['cupy'] = {'installed': False}

    # Triton
    try:
        import triton
        packages['triton'] = {
            'installed': True,
            'version': triton.__version__
        }
    except ImportError:
        packages['triton'] = {'installed': False}

    # PyCUDA
    try:
        import pycuda.driver as cuda
        packages['pycuda'] = {
            'installed': True,
            'version': 'Available'
        }
    except ImportError:
        packages['pycuda'] = {'installed': False}

    return packages


def get_system_info() -> Dict:
    """Get system information."""
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'architecture': platform.machine(),
    }


def run_cuda_check(verbose: bool = False) -> Dict:
    """Run comprehensive CUDA environment check."""
    results = {
        'system': get_system_info(),
        'nvidia_smi': {},
        'cuda_toolkit': {},
        'pytorch': {},
        'profiling_tools': {},
        'optional_packages': {},
    }

    print_section("System Information")
    for key, value in results['system'].items():
        print(f"  {key:20s}: {value}")

    # Check nvidia-smi
    print_section("NVIDIA GPU Detection (nvidia-smi)")
    smi_ok, smi_info = check_nvidia_smi()
    results['nvidia_smi'] = smi_info

    if smi_ok and 'gpus' in smi_info:
        for gpu in smi_info['gpus']:
            print(f"  GPU {gpu['index']}:")
            print(f"    Name:              {gpu['name']}")
            print(f"    Driver Version:    {gpu['driver_version']}")
            print(f"    Memory:            {gpu['memory_total_mb']} MB")
            print(f"    Compute Capability: {gpu['compute_capability']}")
    else:
        print(f"  ERROR: {smi_info.get('error', 'Unknown error')}")

    # Check CUDA toolkit
    print_section("CUDA Toolkit")
    toolkit_ok, toolkit_info = check_cuda_toolkit()
    results['cuda_toolkit'] = toolkit_info

    print(f"  NVCC Version:      {toolkit_info.get('nvcc_version', 'N/A')}")
    print(f"  NVCC Path:         {toolkit_info.get('nvcc_path', 'N/A')}")
    print(f"  CUDA_HOME:         {toolkit_info.get('cuda_home', 'N/A')}")

    if verbose and 'cuda_libraries' in toolkit_info:
        print("\n  CUDA Libraries:")
        for lib, path in toolkit_info['cuda_libraries'].items():
            print(f"    {lib:20s}: {path}")

    # Check PyTorch
    print_section("PyTorch CUDA Support")
    pytorch_ok, pytorch_info = check_pytorch()
    results['pytorch'] = pytorch_info

    if pytorch_ok:
        print(f"  PyTorch Version:   {pytorch_info['pytorch_version']}")
        print(f"  CUDA Available:    {pytorch_info['cuda_available']}")
        print(f"  CUDA Version:      {pytorch_info['cuda_version']}")
        print(f"  cuDNN Version:     {pytorch_info['cudnn_version']}")
        print(f"  cuDNN Enabled:     {pytorch_info['cudnn_enabled']}")
        print(f"  Device Count:      {pytorch_info.get('device_count', 0)}")

        if 'devices' in pytorch_info:
            for device in pytorch_info['devices']:
                print(f"\n  Device {device['index']}:")
                print(f"    Name:              {device['name']}")
                print(f"    Compute Capability: {device['compute_capability']}")
                print(f"    Total Memory:      {device['total_memory_gb']:.2f} GB")
                print(f"    Multiprocessors:   {device['multi_processor_count']}")
    else:
        print(f"  ERROR: {pytorch_info.get('error', 'Unknown error')}")

    # Check profiling tools
    print_section("Profiling Tools")
    profiling_tools = check_profiling_tools()
    results['profiling_tools'] = profiling_tools

    print(f"  nsys (Nsight Systems):  {profiling_tools['nsys']}")
    if profiling_tools['nsys'] and verbose:
        print(f"    Version: {profiling_tools.get('nsys_version', 'N/A')}")

    print(f"  ncu (Nsight Compute):    {profiling_tools['ncu']}")
    if profiling_tools['ncu'] and verbose:
        print(f"    Version: {profiling_tools.get('ncu_version', 'N/A')}")

    print(f"  torch.profiler:          {profiling_tools['torch_profiler']}")

    # Check optional packages
    print_section("Optional CUDA Packages")
    optional_packages = check_optional_packages()
    results['optional_packages'] = optional_packages

    for pkg_name, pkg_info in optional_packages.items():
        status = "Installed" if pkg_info['installed'] else "Not Installed"
        print(f"  {pkg_name:15s}: {status}")
        if verbose and pkg_info['installed'] and 'version' in pkg_info:
            print(f"    Version: {pkg_info['version']}")

    # Summary
    print_section("Summary")

    checks = {
        'NVIDIA GPU detected': smi_ok,
        'CUDA toolkit (nvcc)': toolkit_ok,
        'PyTorch CUDA support': pytorch_ok,
        'torch.profiler available': profiling_tools['torch_profiler'],
    }

    all_passed = all(checks.values())

    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {check:30s}: {status}")

    print()
    if all_passed:
        print("  All critical checks passed! CUDA environment is ready.")
    else:
        print("  Some checks failed. Please install missing components.")
        print("\n  Recommendations:")
        if not toolkit_ok:
            print("    - Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
        if not pytorch_ok:
            print("    - Install PyTorch with CUDA: https://pytorch.org/get-started/locally/")
        if not profiling_tools['torch_profiler']:
            print("    - torch.profiler should be available with PyTorch 2.0+")

    print()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Check CUDA environment for MLFF Distiller",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed information'
    )
    parser.add_argument(
        '--export-json',
        type=str,
        metavar='FILE',
        help='Export results to JSON file'
    )

    args = parser.parse_args()

    results = run_cuda_check(verbose=args.verbose)

    if args.export_json:
        with open(args.export_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results exported to: {args.export_json}")

    # Exit with error code if critical checks failed
    if not (results['pytorch'].get('cuda_available', False)):
        sys.exit(1)

    sys.exit(0)


if __name__ == '__main__':
    main()
