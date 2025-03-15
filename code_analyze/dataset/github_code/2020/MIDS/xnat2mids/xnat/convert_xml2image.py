import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np



def xml2image(xml_file):

    tree = ET.parse(str(xml_file))
    root = tree.getroot()

    mki = 0
    values = []
    for imageanotation in root.iter('{gme://caCORE.caCORE/4.4/edu.northwestern.radiology.AIM}ImageAnnotation'):
        value = imageanotation[3].attrib['value']
        for markupentity in imageanotation.iter('{gme://caCORE.caCORE/4.4/edu.northwestern.radiology.AIM}MarkupEntity'):
            mki+=1
            for _2dspatioancoordinate in markupentity.iter('{gme://caCORE.caCORE/4.4/edu.northwestern.radiology.AIM}TwoDimensionSpatialCoordinate'):
                values.append([mki, value, _2dspatioancoordinate[0].attrib['value'], _2dspatioancoordinate[1].attrib['value'], _2dspatioancoordinate[2].attrib['value']])

    coords = pd.DataFrame.from_records(values, columns=['Mark', 'Type', 'idx', 'x', 'y'])

    masks = []
    eq = list(enumerate(list(set(coords['Type'])), 1))
    eq = {x: i for i, x in eq}
    coords['label'] = coords['Type'].map(eq)
    n_diff_marks = len(list(set(coords['label'])))
    for t_mark in list(set(coords['label'])):
        marks = coords[(coords['label'] == t_mark)]
        for m_idx in list(set(marks['Mark'])):
            mark = marks[marks['Mark'] == m_idx]
            print(mark)
            poly = []
            poly.append(mark)
            # La mascara de fondo tiene que estar en np.uint8
            mask = np.zeros(img.shape, dtype=np.uint8)
            # points = np.int32(mark[['x','y']].astype(float).astype("int32").to_numpy())
            test = mark[["x", "y"]].to_numpy()
            # Hay que pasar de string a float
            test = test.astype(np.float32)
            # Despues a int
            test = test.astype(np.int32)
            # Despues meter el array de numpy en un [], sin el [] no funciona
            test = np.array([test], dtype=np.int32)
            cv2.fillPoly(mask, test, 255)
            mask = np.where(mask == 255, t_mark, mask)
            masks.append(mask)

    dest = np.zeros(masks[0].shape + (3,)).astype("uint8")
    for mask in masks:
        a = np.array([mask, mask, mask])

        a = np.transpose(a, (1, 2, 0))
        print(a.shape)
        dest = cv2.addWeighted(dest, 1, a, 1, 0)

    gray_image = cv2.cvtColor(dest, cv2.COLOR_BGR2GRAY)
    np.image.imsave(xml_file.parent.joinpath(xml_file.stem + ".png"), gray_image)
    coords.to_csv(xml_file.parent.joinpath(xml_file.stem + ".csv"), sep='\t')