import cv2


def clarity_rating(image):
    """
    Calculate a clarity rating for an image. The method used is called the variance of the Laplacian.
    A Laplacian kernel is convolved over the image, and then the variance is calculated.
    A low clarity rating indicates a blurry image, a high clarity rating indicates a clear image.
    :param image: the image to calculate a clarity rating for
    :return: the image's clarity rating
    """
    # Read in the image as a grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize the image to make ratings consistent across images and to reduce computation time
    gray_resized = cv2.resize(gray, (200, 200))

    # Calculate the variance of the Laplacian
    rating = cv2.Laplacian(gray_resized, cv2.CV_64F).var()

    return rating
