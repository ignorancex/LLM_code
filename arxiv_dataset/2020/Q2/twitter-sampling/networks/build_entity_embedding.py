import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer


def main():
    timer = Timer()
    timer.start()

    app_name = 'cyberbullying'

    complete_user_id_set = set()
    with open('../data/{0}_out/complete_user_{0}.txt'.format(app_name), 'r') as fin:
        for line in fin:
            tid, root_uid, _ = line.rstrip().split(',', 2)
            complete_user_id_set.add(root_uid)

    embed_uid_dict = {'u{0}'.format(embed): uid for embed, uid in enumerate(sorted(list(complete_user_id_set)))}
    num_user_complete = len(embed_uid_dict)
    print('{0} users appear in the complete set'.format(num_user_complete))

    with open('../networks/{0}_embed_user.txt'.format(app_name), 'w') as fout:
        for uid in sorted(embed_uid_dict.keys()):
            fout.write('{0},{1}\n'.format(uid, embed_uid_dict[uid]))

    print('>>> Finish embedding users')
    timer.stop()

    complete_hashtag_id_set = set()
    with open('../data/{0}_out/complete_hashtag_{0}.txt'.format(app_name), 'r', encoding='utf-8') as fin:
        for line in fin:
            tid, *hashtags = line.rstrip().lower().split(',')
            complete_hashtag_id_set.update(hashtags)

    embed_hid_dict = {'h{0}'.format(embed): hashtag for embed, hashtag in enumerate(sorted(list(complete_hashtag_id_set)))}
    num_hashtag_complete = len(embed_hid_dict)
    print('{0} hashtags appear in the complete set'.format(num_hashtag_complete))

    with open('../networks/{0}_embed_hashtag.txt'.format(app_name), 'w', encoding='utf-8') as fout:
        for hid in sorted(embed_hid_dict.keys()):
            fout.write('{0},{1}\n'.format(hid, embed_hid_dict[hid]))

    print('>>> Finish embedding hashtags')
    timer.stop()


if __name__ == '__main__':
    main()
