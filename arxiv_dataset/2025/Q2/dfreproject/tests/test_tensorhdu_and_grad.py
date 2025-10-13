import os
import pytest
import torch
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

@pytest.mark.unit
class TestTensorHDUAndGrad:
    def test_tensorhdu_tensor_and_data(self, source_fits_file):
        # Test that TensorHDU exposes .tensor as a torch.Tensor and .data as a numpy array,
        # and that their values match for a numpy input.
        source_hdu = fits.open(source_fits_file)[0]
        
        if hasattr(source_hdu, 'data') and source_hdu.data is not None:
            source_hdu.data = np.asarray(source_hdu.data, dtype=np.float64).copy()

        from dfreproject.tensorhdu import TensorHDU

        hdu = TensorHDU(data=source_hdu.data, header=fits.getheader(source_fits_file))

        assert isinstance(hdu.tensor, torch.Tensor)
        assert isinstance(hdu.data, np.ndarray)
        
        np.testing.assert_allclose(hdu.tensor.detach().cpu().numpy(), hdu.data)

    def test_tensorhdu_accepts_tensor(self, source_fits_file):
        # Test that TensorHDU accepts a torch.Tensor as input and .tensor/.data values match,
        # using data from source_fits_file.
        from dfreproject.tensorhdu import TensorHDU

        arr = fits.getdata(source_fits_file)
        arr = np.asarray(arr, dtype=np.float64).copy()
        t = torch.tensor(arr, dtype=torch.float64)

        hdu = TensorHDU(data=t, header=fits.getheader(source_fits_file))

        assert isinstance(hdu.tensor, torch.Tensor)
        assert isinstance(hdu.data, np.ndarray)

        np.testing.assert_allclose(hdu.tensor.detach().cpu().numpy(), hdu.data)

    def test_gradient_through_reprojection(self, source_fits_file, target_fits_file, device):
        # Test that gradients propagate through the calculate_reprojection pipeline.
        # This ensures differentiability for PyTorch workflows.
        
        from dfreproject.tensorhdu import TensorHDU
        from dfreproject.reproject import calculate_reprojection

        arr = fits.getdata(source_fits_file)
        arr = np.asarray(arr, dtype=np.float64).copy()
        t = torch.tensor(arr, dtype=torch.float64, requires_grad=True, device=device)

        source_header = fits.getheader(source_fits_file)
        hdu = TensorHDU(data=t, header=source_header)
        target_header = fits.getheader(target_fits_file)
        target_wcs = WCS(target_header)
        out = calculate_reprojection(
            source_hdus=hdu,
            target_wcs=target_wcs,
            order='bilinear',
            device=device,
            requires_grad=True
        )
        assert isinstance(out, torch.Tensor)
        assert out.requires_grad
        # Backpropagate through the sum of the output
        s = out.sum()
        s.backward()
        # Check that the input tensor received gradients
        assert t.grad is not None
        assert t.grad.shape == t.shape
        assert torch.any(t.grad != 0)

    def test_gradient_with_numpy_tuple(self, source_fits_file, target_fits_file, device):
        # Test that calculate_reprojection works and is differentiable when source_hdus is (numpy array, header)
        from dfreproject.reproject import calculate_reprojection
        arr = fits.getdata(source_fits_file)
        arr = np.asarray(arr, dtype=np.float64).copy()
     

        header = fits.getheader(source_fits_file)
        target_header = fits.getheader(target_fits_file)
        target_wcs = WCS(target_header)
        # Pass as (numpy array, header)
        out = calculate_reprojection(
            source_hdus=(arr, header),
            target_wcs=target_wcs,
            order='bilinear',
            device=device,
            requires_grad=True
        )
        assert isinstance(out, torch.Tensor)
        assert out.requires_grad
        # Backpropagate through the sum of the output
        s = out.sum()
        s.backward()
        # No direct tensor to check grad, but should not error

    def test_gradient_with_tensor_tuple(self, source_fits_file, target_fits_file, device):
        # Test that calculate_reprojection works and is differentiable when source_hdus is (tensor, header)
        from dfreproject.reproject import calculate_reprojection
        arr = fits.getdata(source_fits_file)
        arr = np.asarray(arr, dtype=np.float64).copy()
       
        t = torch.tensor(arr, dtype=torch.float64, requires_grad=True, device=device)
        header = fits.getheader(source_fits_file)
        target_header = fits.getheader(target_fits_file)
        target_wcs = WCS(target_header)
        # Pass as (tensor, header)
        out = calculate_reprojection(
            source_hdus=(t, header),
            target_wcs=target_wcs,
            order='bilinear',
            device=device,
            requires_grad=True
        )
        assert isinstance(out, torch.Tensor)
        assert out.requires_grad

        s = out.sum()
        s.backward()

        assert t.grad is not None
        assert t.grad.shape == t.shape
        assert torch.any(t.grad != 0)

    def test_gradient_with_primaryhdu(self, source_fits_file, target_fits_file, device):
        # Test that calculate_reprojection works and is differentiable when source_hdus is a PrimaryHDU
        from dfreproject.reproject import calculate_reprojection
        hdu = fits.open(source_fits_file)[0]
        if hasattr(hdu, 'data') and hdu.data is not None:
            hdu.data = np.asarray(hdu.data, dtype=np.float64).copy()
     
        target_header = fits.getheader(target_fits_file)
        target_wcs = WCS(target_header)
        # Pass as PrimaryHDU
        out = calculate_reprojection(
            source_hdus=hdu,
            target_wcs=target_wcs,
            order='bilinear',
            device=device,
            requires_grad=True
        )
        assert isinstance(out, torch.Tensor)
        assert out.requires_grad
        s = out.sum()
        s.backward()
        # No direct tensor to check grad, but should not error
