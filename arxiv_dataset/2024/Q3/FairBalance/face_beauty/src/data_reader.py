import pandas as pd
import tensorflow as tf


def load_scut(rating_cols = ["Average"]):

    def retrievePixels(path):
        # img = tf.keras.utils.load_img("../data/images/"+path, grayscale=False)
        img = tf.keras.utils.load_img("../data/images/" + path, target_size=(224, 224), grayscale=False)
        x = tf.keras.utils.img_to_array(img)
        return x

    data0 = pd.read_csv('../data/Ratings.csv')
    data = pd.DataFrame({"Filename": data0["Filename"]})
    # discretize ratings (>3):
    for col in rating_cols:
        data[col] = data0[col].apply(lambda x: 1 if x > 3.0 else 0)

    # extract sensitive attributes (Male=1, Female=0, Asian=1, Caucasian=0)
    sex = []
    race = []
    for file in data["Filename"]:
        if file[0]=='A':
            race.append(1)
        else:
            race.append(0)
        if file[1]=='M':
            sex.append(1)
        else:
            sex.append(0)
    protected = ['sex', 'race']
    data['sex'] = sex
    data['race'] = race
    data['pixels'] = data['Filename'].apply(retrievePixels)
    return data, protected

