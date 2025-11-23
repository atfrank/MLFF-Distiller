"""
Unit tests for CUDA device utilities.

Tests device detection, memory management, and GPU utility functions.
"""

import pytest
import torch

from src.cuda.device_utils import (
    GPUInfo,
    check_memory_leak,
    cuda_memory_tracker,
    empty_cache,
    get_cuda_stream,
    get_device,
    get_gpu_info,
    get_gpu_memory_info,
    get_peak_memory_allocated,
    list_available_devices,
    reset_peak_memory_stats,
    synchronize_device,
    torch_device_context,
    warmup_cuda,
)


class TestGetDevice:
    """Test device selection and fallback."""

    def test_get_device_default_cpu(self):
        """Test default device selection when CUDA not preferred."""
        device = get_device(prefer_cuda=False)
        assert device.type == 'cpu'

    def test_get_device_explicit_cpu(self):
        """Test explicit CPU device."""
        device = get_device(device='cpu')
        assert device.type == 'cpu'

    def test_get_device_explicit_torch_device(self):
        """Test passing torch.device object."""
        cpu_device = torch.device('cpu')
        device = get_device(device=cpu_device)
        assert device.type == 'cpu'
        assert device == cpu_device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_device_cuda_available(self):
        """Test CUDA device selection when available."""
        device = get_device(prefer_cuda=True)
        assert device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_device_specific_gpu(self):
        """Test selecting specific GPU index."""
        device = get_device(gpu_index=0)
        assert device.type == 'cuda'
        assert device.index == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_device_invalid_gpu_index(self):
        """Test invalid GPU index falls back to GPU 0."""
        device = get_device(gpu_index=999)
        assert device.type == 'cuda'
        assert device.index == 0


class TestGPUInfo:
    """Test GPU information retrieval."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_gpu_info(self):
        """Test retrieving GPU information."""
        info = get_gpu_info(0)

        assert isinstance(info, GPUInfo)
        assert info.index == 0
        assert len(info.name) > 0
        assert info.compute_capability[0] > 0
        assert info.total_memory_gb > 0
        assert info.multi_processor_count > 0
        assert info.warp_size == 32  # Standard for NVIDIA GPUs

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_info_properties(self):
        """Test GPUInfo computed properties."""
        info = get_gpu_info(0)

        # Test compute capability string
        cap_str = info.compute_capability_str
        assert '.' in cap_str
        assert float(cap_str) > 0

        # Test string representation
        info_str = str(info)
        assert info.name in info_str
        assert 'Compute Capability' in info_str

    def test_get_gpu_info_no_cuda(self):
        """Test GPU info when CUDA not available."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, can't test failure case")

        with pytest.raises(RuntimeError, match="CUDA is not available"):
            get_gpu_info(0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_gpu_info_invalid_index(self):
        """Test invalid GPU index."""
        with pytest.raises(RuntimeError, match="out of range"):
            get_gpu_info(999)


class TestMemoryManagement:
    """Test memory monitoring and management."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_gpu_memory_info(self):
        """Test memory info retrieval."""
        mem_info = get_gpu_memory_info()

        assert isinstance(mem_info, dict)
        assert 'allocated' in mem_info
        assert 'reserved' in mem_info
        assert 'free' in mem_info
        assert 'total' in mem_info
        assert 'utilization' in mem_info

        # All values should be non-negative
        for value in mem_info.values():
            assert value >= 0

        # Total should be positive
        assert mem_info['total'] > 0

    def test_get_gpu_memory_info_no_cuda(self):
        """Test memory info when CUDA not available."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, can't test failure case")

        mem_info = get_gpu_memory_info()
        assert mem_info['total'] == 0.0
        assert mem_info['allocated'] == 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_tracking(self):
        """Test memory allocation tracking."""
        initial_mem = get_gpu_memory_info()['allocated']

        # Allocate tensor
        tensor = torch.randn(1000, 1000, device='cuda')

        current_mem = get_gpu_memory_info()['allocated']
        assert current_mem > initial_mem

        # Free tensor
        del tensor
        empty_cache()

        # Memory may not decrease immediately due to caching
        # but it should not increase further
        final_mem = get_gpu_memory_info()['allocated']
        assert final_mem <= current_mem

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_peak_memory_tracking(self):
        """Test peak memory statistics."""
        reset_peak_memory_stats()
        initial_peak = get_peak_memory_allocated()

        # Allocate large tensor
        tensor = torch.randn(5000, 5000, device='cuda')

        peak_mem = get_peak_memory_allocated()
        assert peak_mem > initial_peak

        del tensor
        empty_cache()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_leak_detection(self):
        """Test memory leak detection."""
        empty_cache()

        initial_mem = get_gpu_memory_info()['allocated']

        # Simulate work without leak
        for _ in range(10):
            tensor = torch.randn(100, 100, device='cuda')
            _ = tensor.sum()
            del tensor

        synchronize_device()

        # Should not leak
        no_leak = check_memory_leak(initial_mem, tolerance_mb=5.0)
        assert no_leak

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_memory_tracker_context(self):
        """Test CUDA memory tracker context manager."""
        with cuda_memory_tracker(name="Test"):
            tensor = torch.randn(1000, 1000, device='cuda')
            _ = tensor.sum()

        # Context manager should complete without error


class TestDeviceUtilities:
    """Test device utility functions."""

    def test_list_available_devices(self):
        """Test listing available devices."""
        devices = list_available_devices()

        assert isinstance(devices, list)
        assert 'cpu' in devices

        if torch.cuda.is_available():
            assert any('cuda' in d for d in devices)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_synchronize_device(self):
        """Test device synchronization."""
        tensor = torch.randn(100, 100, device='cuda')
        _ = tensor.sum()

        # Should not raise error
        synchronize_device('cuda')

    def test_synchronize_device_cpu(self):
        """Test synchronization on CPU (should be no-op)."""
        synchronize_device('cpu')

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_warmup_cuda(self):
        """Test CUDA warmup."""
        # Should not raise error
        warmup_cuda(n_iterations=5)

    def test_warmup_cuda_no_device(self):
        """Test warmup when CUDA not available."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, can't test failure case")

        # Should not raise error, just return
        warmup_cuda()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_cuda_stream(self):
        """Test CUDA stream creation."""
        stream = get_cuda_stream()
        assert stream is not None
        assert isinstance(stream, torch.cuda.Stream)

    def test_get_cuda_stream_no_cuda(self):
        """Test stream creation when CUDA not available."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, can't test failure case")

        stream = get_cuda_stream()
        assert stream is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_torch_device_context(self):
        """Test device context manager."""
        with torch_device_context('cuda:0'):
            # Operations in this context
            tensor = torch.randn(10, 10, device='cuda')
            assert tensor.is_cuda

    def test_torch_device_context_cpu(self):
        """Test device context with CPU."""
        with torch_device_context('cpu'):
            tensor = torch.randn(10, 10)
            assert not tensor.is_cuda


class TestEmptyCache:
    """Test cache management."""

    def test_empty_cache(self):
        """Test empty cache function."""
        # Should not raise error
        empty_cache()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_empty_cache_cuda(self):
        """Test cache emptying with CUDA."""
        # Allocate and free memory
        tensor = torch.randn(1000, 1000, device='cuda')
        del tensor

        # Empty cache
        empty_cache()

        # Should reduce reserved memory
        mem_info = get_gpu_memory_info()
        # Just check it doesn't error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
