"""Parse Annotations

Read COCO (JSON) and TUHOI (CSV) annotations files and combine them in
a single CSV.
"""
import json
import pandas as pd


def parse_tuhoi(path):
    """
    Read TUHOI annotations and generate object annotations and captions
    in the form: 'subject-verb-object'.

    Args:
        path: TUHOI annotations file path

    Returns:
        A dataframe with columns: 'image_name', 'object', caption'
    """
    tuhoi_df = pd.read_csv(path, encoding='ISO-8859-1')

    new_df = pd.DataFrame(columns=['image_id', 'object', 'caption'])

    for _, row in enumerate(tuhoi_df.itertuples()):
        for tag_index in range(1, 10):
            verbs = getattr(row, 'tag' + str(tag_index)).split('\n')
            if ''.join(verbs) == '':
                break
            for verb in verbs:
                if verb != 'n' and verb != '':
                    caption = 'person ' + verb + ' ' + getattr(row, 'orig_tag' + str(tag_index))
                    object_category = getattr(row, 'orig_tag' + str(tag_index))
                    new_row = pd.DataFrame({'image_id': getattr(row, 'image_name'),
                                            'object': object_category,
                                            'caption': caption}, index=[0])
                    new_df = pd.concat([new_df, new_row])
    return new_df


def parse_coco(path):
    """
    Convert COCO annotations from JSON to DataFrame.

    Args:
        path: COCO annotations file path

    Returns:
        A DataFrame with a column for each dictionary key
    """
    fcoco = open(path)
    coco_dictionary = json.load(fcoco)
    return pd.DataFrame(coco_dictionary.get('annotations'))


def append_row(caption_df, category_df, img_id, id_prefix, new_df):
    """
    Extract the caption and the category related to 'img_id' from
    'caption_df' and append them as a row to 'new_df'.

    Args:
        caption_df: The caption dataframe
        category_df: The category dataframe
        img_id: The id of image that will be appended
        id_prefix: The prefix that will be prepended to img_id
            (to discriminate COCO train/val)
        new_df: The Dataframe where the row will be appended

    Returns:
        'new_df' with 'img_id' row appended
    """
    row = caption_df[caption_df['image_id'] == img_id]
    if category_df is None:  # For TUHOI
        category_row = row.object
    else:
        category_row = category_df[category_df.index == img_id].iloc[0]

    if id_prefix is not None:
        new_img_id = id_prefix + str(img_id)
    else:
        new_img_id = str(img_id)

    new_row = pd.DataFrame({'image_id': new_img_id, 'object': category_row, 'caption': row.caption})
    return pd.concat([new_df, new_row])


def parse_gold():
    """
    Read GOLD annotations and combine them. The resulting dataframe
    is written to 'generated/verse_annotations.csv'.

    COCO files 'captions_train2014.json', 'captions_val2014.json',
    'instances_train2014.json', 'instances_val2014.json' must be
    located in 'data/annotations/COCO' directory. TUHOI file
    'crowdflower_result.csv' must be placed in the directory
    'data/annotations/TUHOI'. Category file 'coco-labels-paper.txt'
    must be placed in 'data/labels' folder.
    Labels file '3.5k_verse_gold_image_sense_annotations.csv' must be
    placed in the directory 'data/labels'.
    """
    print('Parsing COCO...')
    caption_train_df = parse_coco('data/annotations/COCO/captions_train2014.json')
    caption_val_df = parse_coco('data/annotations/COCO/captions_val2014.json')
    object_train_df = parse_coco('data/annotations/COCO/instances_train2014.json')
    object_val_df = parse_coco('data/annotations/COCO/instances_val2014.json')

    categories_labels = [line.rstrip('\n') for line in open('data/labels/coco-labels-paper.txt')]
    #  Category IDs are not aligned with the extended COCO version.
    categories_labels = {str(i + 1): categories_labels[i] for i in range(len(categories_labels))}

    object_train_df['category_id'] = object_train_df['category_id'].apply(
        lambda id: categories_labels.get(str(id)))
    object_val_df['category_id'] = object_val_df['category_id'].apply(
        lambda id: categories_labels.get(str(id)))

    object_train_df = object_train_df.groupby('image_id')['category_id'].apply(
        lambda x: ' '.join(list(set(x))))
    object_val_df = object_val_df.groupby('image_id')['category_id'].apply(
        lambda x: ' '.join(list(set(x))))

    print('Parsing TUHOI...')
    tuhoi_df = parse_tuhoi('data/annotations/TUHOI/crowdflower_result.csv')

    labels = pd.read_csv('data/labels/3.5k_verse_gold_image_sense_annotations.csv')
    img_list = labels['image'].unique().tolist()

    new_df1 = pd.DataFrame(columns=['image_id', 'object', 'caption'])
    new_df2 = pd.DataFrame(columns=['image_id', 'object', 'caption'])
    new_df3 = pd.DataFrame(columns=['image_id', 'object', 'caption'])

    print('Aggregating...')
    for img_name in img_list:
        if img_name.startswith('COCO_train2014_'):
            img_id = int(img_name.split('.')[0][len('COCO_train2014_'):])
            new_df1 = append_row(caption_train_df, object_train_df,
                                 img_id, 'COCO_train2014_', new_df1)

        elif img_name.startswith('COCO_val2014_'):
            img_id = int(img_name.split('.')[0][len('COCO_val2014_'):])
            new_df2 = append_row(caption_val_df, object_val_df, img_id, 'COCO_val2014_', new_df2)
        else:
            new_df3 = append_row(tuhoi_df, None, img_name.split('.')[0], None, new_df3)

    new_df = pd.concat([new_df1, new_df2, new_df3])
    new_df.reset_index(drop=True, inplace=True)

    print(len(new_df.image_id), 'rows.')
    print(len(new_df.image_id.unique()), 'unique annotations.')
    print('Writing...')
    new_df.to_csv('generated/verse_annotations.csv', index=False)


def parse_pred():
    """
    Read PRED annotations and combine them. The resulting dataframe
    is written to 'generated/pred_verse_annotations.csv'.

    The files generated with NeuralTalk: 'neuraltalk_run1.json',
    'neuraltalk_run2.json', 'neuraltalk_run3.json' must be placed in
    'pred' folder.
    """
    pred_captions_1 = json.load(open('pred/neuraltalk_run1.json'))
    pred_captions_2 = json.load(open('pred/neuraltalk_run2.json'))
    pred_captions_3 = json.load(open('pred/neuraltalk_run3.json'))

    pred_captions = pred_captions_1 + pred_captions_2 + pred_captions_3
    image_id = list(map(lambda x: x.get('file_name'), pred_captions))
    image_id = list(map(lambda s: s.split('/')[-1], image_id))
    captions = list(map(lambda x: x.get('caption'), pred_captions))

    captions_df = pd.DataFrame({'image_id': image_id, 'caption': captions})
    captions_df = captions_df.groupby('image_id')['caption'].apply(lambda x: "%s" % ', '.join(x))

    object_df = pd.read_pickle('pred/pred_objects.pkl')
    object_df['object'] = object_df['object'].apply(lambda r: str(r[0]))

    pred_df = pd.concat([captions_df, object_df], axis=1)

    pred_df.to_csv('generated/pred_verse_annotations.csv')

if __name__ == '__main__':
    parse_gold()
    parse_pred()
