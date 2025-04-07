from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
from SinGAN.imresize import imresize_to_shape
import SinGAN.functions as functions
import cv2
from PIL import Image


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='/path/to/Input/Images')
    parser.add_argument('--input_name', help='training image name', required=True)
    parser.add_argument('--ref_dir', help='input reference dir', default='/path/to/outputs_fgbg/mask_ObjMST/')
    parser.add_argument('--ref_name', help='reference image name', required=True)
    parser.add_argument('--harmonization_start_scale', help='harmonization injection scale', type=int, required=True)
    parser.add_argument('--mode', help='task to be done', default='harmonization')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    inp_name=opt.input_name
    ref_name=opt.ref_name
    filefolder, filename=os.path.split(ref_name)
    # print("folder=", filefolder, "filename=", filename)
    # filename = os.path.basename(ref_name)
    # print("input_name=", inp_name, "ref_name=", filename)

    if dir2save is None:
        print('task does not exist')
    #elif (os.path.exists(dir2save)):
    #    print("output already exist")
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        real = functions.adjust_scales2image(real, opt)
        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
        if (opt.harmonization_start_scale < 1) | (opt.harmonization_start_scale > (len(Gs)-1)):
            print("injection scale should be between 1 and %d" % (len(Gs)-1))
        else:
            ref = functions.read_image_dir('%s/%s' % (opt.ref_dir, opt.ref_name), opt)
            mask = functions.read_image_dir('%s/%s_mask%s' % (opt.ref_dir,opt.ref_name[:-4],opt.ref_name[-4:]), opt)
            # mask = functions.read_image_dir('%s/%s%s' % (opt.ref_dir,opt.ref_name[:-4],opt.ref_name[-4:]), opt)
            # print("mask_gray.shape=", mask_gray )
            # mask = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2RGB)
            # print("mask.shape=", mask.shape)
            if ref.shape[3] != real.shape[3]:
                mask = imresize_to_shape(mask, [real.shape[2], real.shape[3]], opt)
                mask = mask[:, :, :real.shape[2], :real.shape[3]]
                ref = imresize_to_shape(ref, [real.shape[2], real.shape[3]], opt)
                ref = ref[:, :, :real.shape[2], :real.shape[3]]
            mask = functions.dilate_mask(mask, opt)

            N = len(reals) - 1
            n = opt.harmonization_start_scale
            in_s = imresize(ref, pow(opt.scale_factor, (N - n + 1)), opt)
            in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
            in_s = imresize(in_s, 1 / opt.scale_factor, opt)
            in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
            out = SinGAN_generate(Gs[n:], Zs[n:], reals, NoiseAmp[n:], opt, in_s, n=n, num_samples=1)
            out = (1-mask)*real+mask*out
            # print("dir2save=", dir2save)
            sty_op="/path/to/outputs_fgbg/mask_ObjMST/"+ filefolder+ "/" + inp_name[:-4] + "_" + filename
            plt.imsave("/path/to/outputs_fgbg/mask_ObjMST/"+ filefolder+ "/" + inp_name[:-4] + "_" + filename, functions.convert_image_np(out.detach()), vmin=0, vmax=1)

            image = Image.open(sty_op)

            # Resize the image to 256x256
            size = (512, 512)
            resized_image = image.resize(size, Image.ANTIALIAS)

            # Get the original filename without extension
            file_name, ext = os.path.splitext(os.path.basename(sty_op))

            # Append '_256x256' to the filename
            new_file_name = f"{file_name}_512x512{ext}"

            # Save the resized image with the new filename
            resized_image.save(new_file_name)



            # plt.imsave('%s/start_scale=%d.png' % (dir2save,opt.harmonization_start_scale), functions.convert_image_np(out.detach()), vmin=0, vmax=1)




