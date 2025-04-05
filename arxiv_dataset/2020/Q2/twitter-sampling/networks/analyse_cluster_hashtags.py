import pickle


def main():
    app_name = 'cyberbullying'
    n_cluster = 6

    uid_user_dict = {}
    with open('../networks/{0}_embed_user.txt'.format(app_name), 'r') as fin:
        for line in fin:
            uid, user = line.rstrip().split(',')
            uid_user_dict[uid] = user

    hid_hashtag_dict = {}
    with open('../networks/{0}_embed_hashtag.txt'.format(app_name), 'r', encoding='utf-8') as fin:
        for line in fin:
            hid, hashtag = line.rstrip().split(',')
            hid_hashtag_dict[hid] = hashtag

    for date_type in ['sample', 'complete']:
        hid_uid_stats = pickle.load(open('./{0}_hid_uid_stats.p'.format(date_type), 'rb'))
        for i in range(n_cluster):
            hashtag_cnt = {}
            with open('{0}_cluster{1}.txt'.format(date_type, i), 'r') as fin:
                nodes = fin.readline().split(',')
                uid_set = set([node for node in nodes if node.startswith('u')])
                hid_set = set([node for node in nodes if node.startswith('h')])
                print('in {0}_cluster{1}, {2} users, {3} hashtags'.format(date_type, i, len(uid_set), len(hid_set)))

                for hid in hid_set:
                    temp_dict = {uid: cnt for uid, cnt in hid_uid_stats[hid][1:]}
                    intersect_users = set(temp_dict.keys()).intersection(uid_set)
                    hashtag_cnt[hid] = sum([temp_dict[uid] for uid in intersect_users])

                most_popular_hashtags = sorted(hashtag_cnt.items(), key=lambda x: x[1], reverse=True)[:5]
                decoded_most_popular_hashtags = [(hid_hashtag_dict[x[0]], x[1]) for x in most_popular_hashtags]
                print('most popular hashtags', decoded_most_popular_hashtags)


if __name__ == '__main__':
    main()
