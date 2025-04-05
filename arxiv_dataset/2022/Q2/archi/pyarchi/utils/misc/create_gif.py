import imageio
import os


def create_gif(input_path, output_path, gif_name, duration = 0.5):
    """
    Creates a gif with the images inside input_path and saves it to output_path with
    gif_name as name.

    Parameters
    ------------------
    input_path:
        Path to the folder where all images are stored

    output_path:
        Path to store the gif
    
    gif_name:
        name of the gif

    duration:
        "Length" of the gif


    """

    images = []
    filenames = [f"{i}.png" for i in range(len(os.listdir(input_path)))]
    for filename in filenames:
        images.append(imageio.imread(os.path.join(input_path, filename)))

    kargs = {'duration': duration}

    if os.path.exists(output_path + gif_name):
        print(' Gif already exists. DO you want to proceed (y/n) ?')
        answer = input()

        if answer.lower() == 'y' or answer.lower() == 'yes':
            imageio.mimsave(output_path + gif_name, images, **kargs)
            [os.remove(input_path + f_name) for f_name in filenames]
        else:
            return
    else:
        imageio.mimsave(os.path.join(output_path, gif_name), images, **kargs)


if __name__ == '__main__':
    input_path = '/home/amiguel/archi/sergio_data/'
    output_path = '/home/amiguel/archi/sergio_data/'
    gif_name = 'masks_shape.gif'

    create_gif(input_path, output_path, gif_name)
