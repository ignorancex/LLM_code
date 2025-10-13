from astropy.io.fits import PrimaryHDU
import torch
import numpy as np



class TensorHDU(PrimaryHDU):

    def __init__(
        self,
        data=None,
        header=None,
        do_not_scale_image_data=False,
        ignore_blank=False,
        uint=True,
        scale_back=None,
    ):
        """
        Construct a pytorch tensor HDU (Child class of PrimaryHDU with added tensor property).

        Parameters
        ----------
        data : Pytorch tensor, array or ``astropy.io.fits.hdu.base.DELAYED``, optional
            The data in the HDU.

        header : `~astropy.io.fits.Header`, optional
            The header to be used (as a template). If ``header`` is `None`, a
            minimal header will be provided.

        do_not_scale_image_data : bool, optional
            If `True`, image data is not scaled using BSCALE/BZERO values
            when read. (default: False)

        ignore_blank : bool, optional
            If `True`, the BLANK header keyword will be ignored if present.
            Otherwise, pixels equal to this value will be replaced with
            NaNs. (default: False)

        uint : bool, optional
            Interpret signed integer data where ``BZERO`` is the
            central value and ``BSCALE == 1`` as unsigned integer
            data.  For example, ``int16`` data with ``BZERO = 32768``
            and ``BSCALE = 1`` would be treated as ``uint16`` data.
            (default: True)

        scale_back : bool, optional
            If `True`, when saving changes to a file that contained scaled
            image data, restore the data to the original type and reapply the
            original BSCALE/BZERO values.  This could lead to loss of accuracy
            if scaling back to integer values after performing floating point
            operations on the data.  Pseudo-unsigned integers are automatically
            rescaled unless scale_back is explicitly set to `False`.
            (default: None)
        """
        self.tensor = data
        if data is not None and isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        super().__init__(
            data=data,
            header=header,
            do_not_scale_image_data=do_not_scale_image_data,
            uint=uint,
            ignore_blank=ignore_blank,
            scale_back=scale_back,
        )

    @property
    def tensor(self) -> torch.Tensor:
        """
        Returns the image data as a torch.Tensor.
        """

        return self.__dict__.get("tensor", None)


    @tensor.setter
    def tensor(self, data):
        # Accept torch.Tensor or numpy array, but always store as torch.Tensor
        if data is not None and not isinstance(data, torch.Tensor):
            try:
                if isinstance(data, np.ndarray):
                    data = torch.tensor(data, requires_grad = True)
                    print("Converted numpy array to torch tensor with requires_grad=True.")
                else:
                    data = torch.tensor(data, requires_grad = True)
            except Exception:
                data = torch.tensor(data, requires_grad = True)
        elif isinstance(data, torch.Tensor):
            if data.dtype != torch.float64:
                data = data.to(torch.float64)



        self.__dict__["tensor"] = data

