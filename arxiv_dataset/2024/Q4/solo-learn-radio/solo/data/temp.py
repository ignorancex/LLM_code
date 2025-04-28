class AddNormChannels:
    def __init__(self):
        """Add 3 normalized channels as a callable object.

        Args:None
        """

    def __call__(self, img: Image) -> Image:
        """Applies gaussian blur to an input image.

        Args:
            img (Image): a single channel image in npy format

        Returns:
            Image: blurred image.
        """

        ch1_stretches = [PowerStretch(2), SquaredStretch(), SinhStretch()]
        ch1_stretch = random.choices(ch1_stretches)

        ch2_stretch = LinearStretch(slope=1, intercept=0)

        ch3_stretches = [LogStretch(a=1000), SqrtStretch(), AsinhStretch()]
        ch3_stretch = random.choices(ch3_stretches)

        transform_ch1 =  ch1_stretch[0] + MinMaxInterval()
        transform_ch2 =  ch2_stretch + MinMaxInterval()
        transform_ch3=  ch3_stretch[0] + MinMaxInterval()

        stacked_image = np.dstack((transform_ch1(img), transform_ch2(img), transform_ch3(img)))
        stacked_image_pil = Image.fromarray(np.uint8(stacked_image*255))
        """
        if any(np.isnan(return_image_pil)):
            print("NAN")
        """
        return stacked_image_pil