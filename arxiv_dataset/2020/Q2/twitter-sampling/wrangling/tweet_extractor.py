# -*- coding: utf-8 -*-

""" Extract tweet status from full tweet.
1. tweet_id_str, created_at, timestamp_ms, user_id_str,
   original_lang, retweeted_lang, quoted_lang,
   original_vids, retweeted_vids, quoted_vids,
   original_mentions, retweeted_mentions, quoted_mentions,
   original_hashtags, retweeted_hashtags, quoted_hashtags,
   original_geoname, retweeted_geoname, quoted_geoname,
   original_countrycode, retweeted_countrycode, quoted_countrycode,
   original_filter, retweeted_filter, quoted_filter,
   original_retweet_count, retweeted_retweet_count, quoted_retweet_count,
   original_favorite_count, retweeted_favorite_count, quoted_favorite_count,
   original_user_followers_count, retweeted_user_followers_count, quoted_user_followers_count,
   original_user_friends_count, retweeted_user_friends_count, quoted_user_friends_count,
   original_user_statuses_count, retweeted_user_statuses_count, quoted_user_statuses_count,
   original_user_favourites_count, retweeted_user_favourites_count, quoted_user_favourites_count,
   reply_tweet_id_str, retweeted_tweet_id_str, quoted_tweet_id_str,
   reply_user_id_str, retweeted_user_id_str, quoted_user_id_str,
   original_text, retweeted_text, quoted_text
2. user_id_str, screen_name, created_at, verified, location, followers_count, friends_count, listed_count, statuses_count, description
3. ratemsg, timestamp_ms, track
"""

import sys, os, bz2, re, json, logging, logging.config
from datetime import datetime
from multiprocessing import Process, Queue

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import strify, str2obj, obj2str


class TweetExtractor(object):
    """ Tweet Object Extractor Class.

    :param input_dir: directory that contains all tweet bz2 files
    :param output_dir: directory that composes of 2 folders -- tweet_stats and user_stats

    For a tweet, the dictionaries must include the following fields:

    id:               The integer representation of the unique identifier for this Tweet.
    ******
    entities:         Entities provide structured data from Tweets including resolved URLs, media, hashtags
                      and mentions without having to parse the text to extract that information.
                      ******
                      # We only care about urls information at this moment.
                      urls:       Optional. The URL of the video file
                                  Potential fields:
                                  * url            The t.co URL that was extracted from the Tweet text
                                  * expanded_url   The resolved URL
                                  * display_url	   Not a valid URL but a string to display instead of the URL
                                  * indices	       The character positions the URL was extracted from
                      ******
    ******
    retweeted_status: entities: urls: expanded_url
                      extended_tweet: entities: urls: expanded_url
    ******
    quoted_status:    entities: urls: expanded_url
                      extended_tweet: entities: urls: expanded_url
    """

    def __init__(self, input_dir, output_dir, proc_num=1):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.proc_num = proc_num
        self.logger = None

        self.tweet_stats_path = "{0}/{1}".format(output_dir, 'tweet_stats')
        self.user_stats_path = "{0}/{1}".format(output_dir, 'user_stats')
        os.makedirs(self.tweet_stats_path, exist_ok=True)
        os.makedirs(self.user_stats_path, exist_ok=True)

        self._setup_logger('tweetextractor')

    def set_proc_num(self, n):
        """Set the number of processes used in extracting."""
        self.proc_num = n

    def _setup_logger(self, logger_name):
        """Set logger from conf file."""
        log_dir = '../log/'
        os.makedirs(log_dir, exist_ok=True)
        logging.config.fileConfig('../conf/logging.conf')
        self.logger = logging.getLogger(logger_name)

    @staticmethod
    def _replace_with_nan(obj):
        if obj is None or len(obj) == 0:
            return 'N'
        else:
            return obj

    @staticmethod
    def _replace_comma_space(text):
        return re.sub(',\\s*|\s+', ' ', text)

    def extract(self):
        self.logger.debug('**> Start extracting tweet status from tweet bz2 files...')

        processes = []
        filequeue = Queue()

        for subdir, _, files in os.walk(self.input_dir):
            for f in sorted(files):
                filename, filetype = f.split('.')
                if filetype == 'bz2':
                    filepath = os.path.join(subdir, f)
                    filequeue.put(filepath)

        for w in range(self.proc_num):
            p = Process(target=self._extract_tweet, args=(filequeue,))
            p.daemon = True
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        self.logger.debug('**> Finish extracting tweet status from tweet bz2 files.')

    @staticmethod
    def _extract_vid_from_expanded_url(expanded_url):
        if 'watch?' in expanded_url and 'v=' in expanded_url:
            vid = expanded_url.split('v=')[1][:11]
        elif 'youtu.be' in expanded_url:
            vid = expanded_url.rsplit('/', 1)[-1][:11]
        else:
            return None
        # valid condition: contains only alphanumeric, dash or underline
        valid = re.match('^[\w-]+$', vid) is not None
        if valid and len(vid) == 11:
            return vid
        return None

    def _expanded_urls(self, urls):
        expanded_urls = []
        for url in urls:
            if url['expanded_url'] is not None:
                expanded_urls.append(url['expanded_url'])

        vids = set()
        for expanded_url in expanded_urls:
            vid = self._extract_vid_from_expanded_url(expanded_url)
            if vid is not None:
                vids.add(vid)
        return vids

    def _extract_vids(self, tweet):
        original_urls = []
        retweeted_urls = []
        quoted_urls = []
        try:
            original_urls.extend(tweet['entities']['urls'])
        except KeyError:
            pass
        try:
            retweeted_urls.extend(tweet['retweeted_status']['entities']['urls'])
        except KeyError:
            pass
        try:
            retweeted_urls.extend(tweet['retweeted_status']['extended_tweet']['entities']['urls'])
        except KeyError:
            pass
        try:
            quoted_urls.extend(tweet['quoted_status']['entities']['urls'])
        except KeyError:
            pass
        try:
            quoted_urls.extend(tweet['quoted_status']['extended_tweet']['entities']['urls'])
        except KeyError:
            pass

        original_vids = self._replace_with_nan(self._expanded_urls(original_urls))
        retweeted_vids = self._replace_with_nan(self._expanded_urls(retweeted_urls))
        quoted_vids = self._replace_with_nan(self._expanded_urls(quoted_urls))
        return original_vids, retweeted_vids, quoted_vids

    def _extract_hashtags(self, tweet):
        original_hashtags = set()
        retweeted_hashtags = set()
        quoted_hashtags = set()
        try:
            for hashtag in tweet['entities']['hashtags']:
                original_hashtags.add(hashtag['text'])
        except KeyError:
            pass
        try:
            for hashtag in tweet['retweeted_status']['entities']['hashtags']:
                retweeted_hashtags.add(hashtag['text'])
        except KeyError:
            pass
        try:
            for hashtag in tweet['retweeted_status']['extended_tweet']['entities']['hashtags']:
                retweeted_hashtags.add(hashtag['text'])
        except KeyError:
            pass
        try:
            for hashtag in tweet['quoted_status']['entities']['hashtags']:
                quoted_hashtags.add(hashtag['text'])
        except KeyError:
            pass
        try:
            for hashtag in tweet['quoted_status']['extended_tweet']['entities']['hashtags']:
                quoted_hashtags.add(hashtag['text'])
        except KeyError:
            pass
        original_hashtags = self._replace_with_nan(original_hashtags)
        retweeted_hashtags = self._replace_with_nan(retweeted_hashtags)
        quoted_hashtags = self._replace_with_nan(quoted_hashtags)
        return original_hashtags, retweeted_hashtags, quoted_hashtags

    def _extract_mentions(self, tweet):
        original_mentions = set()
        retweeted_mentions = set()
        quoted_mentions = set()
        try:
            for user_mention in tweet['entities']['user_mentions']:
                if user_mention['id_str'] is not None:
                    original_mentions.add(user_mention['id_str'])
        except KeyError:
            pass
        try:
            for user_mention in tweet['retweeted_status']['entities']['user_mentions']:
                if user_mention['id_str'] is not None:
                    retweeted_mentions.add(user_mention['id_str'])
        except KeyError:
            pass
        try:
            for user_mention in tweet['retweeted_status']['extended_tweet']['entities']['user_mentions']:
                if user_mention['id_str'] is not None:
                    retweeted_mentions.add(user_mention['id_str'])
        except KeyError:
            pass
        try:
            for user_mention in tweet['quoted_status']['entities']['user_mentions']:
                if user_mention['id_str'] is not None:
                    quoted_mentions.add(user_mention['id_str'])
        except KeyError:
            pass
        try:
            for user_mention in tweet['quoted_status']['extended_tweet']['entities']['user_mentions']:
                if user_mention['id_str'] is not None:
                    quoted_mentions.add(user_mention['id_str'])
        except KeyError:
            pass
        original_mentions = self._replace_with_nan(original_mentions)
        retweeted_mentions = self._replace_with_nan(retweeted_mentions)
        quoted_mentions = self._replace_with_nan(quoted_mentions)
        return original_mentions, retweeted_mentions, quoted_mentions

    def _extract_entities(self, tweet, field):
        if field in tweet:
            tweet_id_str = tweet[field]['id_str']
            user_id_str = tweet[field]['user']['id_str']
            user_location = tweet[field]['user']['location']
            if 'lang' in tweet[field]:
                lang = tweet[field]['lang']
            else:
                lang = 'N'

            if tweet[field]['place'] is not None:
                geo = self._replace_comma_space(tweet[field]['place']['full_name'])
                cc = self._replace_comma_space(tweet[field]['place']['country_code'])
            else:
                geo = 'N'
                cc = 'N'
            filter = tweet[field]['filter_level']

            retweet_count = tweet[field]['retweet_count']
            favorite_count = tweet[field]['favorite_count']

            user_followers_count = tweet[field]['user']['followers_count']
            user_friends_count = tweet[field]['user']['friends_count']
            user_statuses_count = tweet[field]['user']['statuses_count']
            user_favourites_count = tweet[field]['user']['favourites_count']

            if 'extended_tweet' in tweet[field] and 'full_text' in tweet[field]['extended_tweet']:
                text = self._replace_comma_space(tweet[field]['extended_tweet']['full_text'])
            else:
                text = self._replace_comma_space(tweet[field]['text'])
        else:
            tweet_id_str = 'N'
            user_id_str = 'N'
            user_location = 'N'
            lang = 'N'

            geo = 'N'
            cc = 'N'
            filter = 'N'

            retweet_count = 'N'
            favorite_count = 'N'

            user_followers_count = 'N'
            user_friends_count = 'N'
            user_statuses_count = 'N'
            user_favourites_count = 'N'

            text = 'N'
        return (tweet_id_str, user_id_str, user_location,
                lang, geo, cc, filter,
                retweet_count, favorite_count,
                user_followers_count, user_friends_count, user_statuses_count, user_favourites_count,
                text)

    def _extract_user_entities(self, user):
        user_screen_name = user['screen_name']
        user_created_at = datetime.strptime(user['created_at'], '%a %b %d %H:%M:%S %z %Y').strftime('%Y-%m-%d')
        user_verified = 'Y' if user['verified'] else 'N'
        user_location = self._replace_comma_space(user['location']) if user['location'] else 'N'
        user_followers_count = user['followers_count']
        user_friends_count = user['friends_count']
        user_listed_count = user['listed_count']
        user_statuses_count = user['statuses_count']
        user_description = self._replace_comma_space(user['description']) if user['description'] else 'N'
        return (user_screen_name, user_created_at, user_verified, user_location,
                user_followers_count, user_friends_count, user_listed_count, user_statuses_count, user_description)

    def _extract_tweet(self, filequeue):
        while not filequeue.empty():
            filepath = filequeue.get()
            try:
                filedata = bz2.BZ2File(filepath, mode='r')
            except:
                self.logger.warn('Exists non-bz2 file {0} in dataset folder'.format(filepath))
                continue
            filename, filetype = os.path.basename(os.path.normpath(filepath)).split('.')

            tweet_output = bz2.open(os.path.join(self.tweet_stats_path, '{0}.bz2'.format(filename)), 'at')
            user_output = bz2.open(os.path.join(self.user_stats_path, '{0}.bz2'.format(filename)), 'at')

            visited_user_ids = set()
            for line in filedata:
                try:
                    if line.rstrip():
                        tweet_json = json.loads(line)

                        # 3. ratemsg, timestamp_ms, track
                        if 'limit' in tweet_json:
                            # rate limit message
                            # {"limit":{"track":283540,"timestamp_ms":"1483189188944"}}
                            tweet_output.write('{0},{1},{2}\n'.format('ratemsg', tweet_json['limit']['timestamp_ms'], tweet_json['limit']['track']))
                            continue

                        if 'id_str' not in tweet_json:
                            continue

                        # 1. tweet_id_str, created_at, timestamp_ms, user_id_str,
                        #    original_lang, retweeted_lang, quoted_lang,
                        #    original_vids, retweeted_vids, quoted_vids,
                        #    original_mentions, retweeted_mentions, quoted_mentions,
                        #    original_hashtags, retweeted_hashtags, quoted_hashtags,
                        #    original_geoname, retweeted_geoname, quoted_geoname,
                        #    original_countrycode, retweeted_countrycode, quoted_countrycode,
                        #    original_filter, retweeted_filter, quoted_filter,
                        #    original_retweet_count, retweeted_retweet_count, quoted_retweet_count,
                        #    original_favorite_count, retweeted_favorite_count, quoted_favorite_count,
                        #    original_user_followers_count, retweeted_user_followers_count, quoted_user_followers_count,
                        #    original_user_friends_count, retweeted_user_friends_count, quoted_user_friends_count,
                        #    original_user_statuses_count, retweeted_user_statuses_count, quoted_user_statuses_count,
                        #    original_user_favourites_count, retweeted_user_favourites_count, quoted_user_favourites_count,
                        #    reply_tweet_id_str, retweeted_tweet_id_str, quoted_tweet_id_str,
                        #    reply_user_id_str, retweeted_user_id_str, quoted_user_id_str,
                        #    original_text, retweeted_text, quoted_text
                        tweet_id = tweet_json['id_str']
                        created_at = obj2str(str2obj(tweet_json['created_at'], fmt='tweet'), fmt='youtube')
                        timestamp_ms = tweet_json['timestamp_ms']
                        user_id_str = tweet_json['user']['id_str']
                        if 'lang' in tweet_json:
                            lang = tweet_json['lang']
                        else:
                            lang = 'N'

                        original_vids, retweeted_vids, quoted_vids = self._extract_vids(tweet_json)
                        original_mentions, retweeted_mentions, quoted_mentions = self._extract_mentions(tweet_json)
                        original_hashtags, retweeted_hashtags, quoted_hashtags, = self._extract_hashtags(tweet_json)

                        if tweet_json['place'] is not None:
                            original_geo = self._replace_comma_space(tweet_json['place']['full_name'])
                            original_cc = self._replace_comma_space(tweet_json['place']['country_code'])
                        else:
                            original_geo = 'N'
                            original_cc = 'N'

                        original_filter = tweet_json['filter_level']

                        original_retweet_count = tweet_json['retweet_count']
                        original_favorite_count = tweet_json['favorite_count']

                        original_user_followers_count = tweet_json['user']['followers_count']
                        original_user_friends_count = tweet_json['user']['friends_count']
                        original_user_statuses_count = tweet_json['user']['statuses_count']
                        original_user_favourites_count = tweet_json['user']['favourites_count']

                        reply_tweet_id_str = self._replace_with_nan(tweet_json['in_reply_to_status_id_str'])
                        reply_user_id_str = self._replace_with_nan(tweet_json['in_reply_to_user_id_str'])

                        if 'extended_tweet' in tweet_json and 'full_text' in tweet_json['extended_tweet']:
                            text = self._replace_comma_space(tweet_json['extended_tweet']['full_text'])
                        elif tweet_json['text'] is not None:
                            text = self._replace_comma_space(tweet_json['text'])
                        else:
                            text = 'N'

                        retweeted_tweet_id_str, retweeted_user_id_str, retweeted_user_location,\
                        retweeted_lang, retweeted_geo, retweeted_cc, retweeted_filter, \
                        retweeted_retweet_count, retweeted_favorite_count, retweeted_user_followers_count, \
                        retweeted_user_friends_count, retweeted_user_statuses_count, retweeted_user_favourites_count, \
                        retweeted_text = self._extract_entities(tweet_json, 'retweeted_status')

                        quoted_tweet_id_str, quoted_user_id_str, quoted_user_location,\
                        quoted_lang, quoted_geo, quoted_cc, quoted_filter, \
                        quoted_retweet_count, quoted_favorite_count, quoted_user_followers_count, \
                        quoted_user_friends_count, quoted_user_statuses_count, quoted_user_favourites_count, \
                        quoted_text = self._extract_entities(tweet_json, 'quoted_status')

                        tweet_output.write('{0},{1},{2},{3},'
                                           '{4},{5},{6},'
                                           '{7},{8},{9},'
                                           '{10},{11},{12},'
                                           '{13},{14},{15},'
                                           '{16},{17},{18},'
                                           '{19},{20},{21},'
                                           '{22},{23},{24},'
                                           '{25},{26},{27},'
                                           '{28},{29},{30},'
                                           '{31},{32},{33},'
                                           '{34},{35},{36},'
                                           '{37},{38},{39},'
                                           '{40},{41},{42},'
                                           '{43},{44},{45},'
                                           '{46},{47},{48},'
                                           '{49},{50},{51}\n'
                                           .format(tweet_id, created_at, timestamp_ms, user_id_str,
                                                   lang, retweeted_lang, quoted_lang,
                                                   strify(original_vids, delimiter=';'), strify(retweeted_vids, delimiter=';'), strify(quoted_vids, delimiter=';'),
                                                   strify(original_mentions, delimiter=';'), strify(retweeted_mentions, delimiter=';'), strify(quoted_mentions, delimiter=';'),
                                                   strify(original_hashtags, delimiter=';'), strify(retweeted_hashtags, delimiter=';'), strify(quoted_hashtags, delimiter=';'),
                                                   original_geo, retweeted_geo, quoted_geo,
                                                   original_cc, retweeted_cc, quoted_cc,
                                                   original_filter, retweeted_filter, quoted_filter,
                                                   original_retweet_count, retweeted_retweet_count, quoted_retweet_count,
                                                   original_favorite_count, retweeted_favorite_count, quoted_favorite_count,
                                                   original_user_followers_count, retweeted_user_followers_count, quoted_user_followers_count,
                                                   original_user_friends_count, retweeted_user_friends_count, quoted_user_friends_count,
                                                   original_user_statuses_count, retweeted_user_statuses_count, quoted_user_statuses_count,
                                                   original_user_favourites_count, retweeted_user_favourites_count, quoted_user_favourites_count,
                                                   reply_tweet_id_str, retweeted_tweet_id_str, quoted_tweet_id_str,
                                                   reply_user_id_str, retweeted_user_id_str, quoted_user_id_str,
                                                   text, retweeted_text, quoted_text))

                        # 2. user_id_str, screen_name, created_at, verified, location, followers_count, friends_count, listed_count, statuses_count, description
                        if user_id_str not in visited_user_ids:
                            user_screen_name, user_created_at, user_verified, user_location, user_followers_count, user_friends_count, user_listed_count, user_statuses_count, user_description = self._extract_user_entities(tweet_json['user'])
                            user_output.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}\n'
                                              .format(user_id_str, user_screen_name, user_created_at, user_verified,
                                                      user_location, user_followers_count, user_friends_count,
                                                      user_listed_count, user_statuses_count, user_description))
                            visited_user_ids.add(user_screen_name)

                        if 'retweeted_status' in tweet_json:
                            ruser_id_str = tweet_json['retweeted_status']['user']['id_str']
                            if ruser_id_str not in visited_user_ids:
                                ruser_screen_name, ruser_created_at, ruser_verified, ruser_location, ruser_followers_count, ruser_friends_count, ruser_listed_count, ruser_statuses_count, ruser_description = self._extract_user_entities(tweet_json['retweeted_status']['user'])
                                user_output.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}\n'
                                                  .format(ruser_id_str, ruser_screen_name, ruser_created_at, ruser_verified,
                                                          ruser_location, ruser_followers_count, ruser_friends_count,
                                                          ruser_listed_count, ruser_statuses_count, ruser_description))
                                visited_user_ids.add(ruser_id_str)

                        if 'quoted_status' in tweet_json:
                            quser_id_str = tweet_json['quoted_status']['user']['id_str']
                            if quser_id_str not in visited_user_ids:
                                quser_screen_name, quser_created_at, quser_verified, quser_location, quser_followers_count, quser_friends_count, quser_listed_count, quser_statuses_count, quser_description = self._extract_user_entities(tweet_json['quoted_status']['user'])
                                user_output.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}\n'
                                                  .format(quser_id_str, quser_screen_name, quser_created_at, quser_verified,
                                                          quser_location, quser_followers_count, quser_friends_count,
                                                          quser_listed_count, quser_statuses_count, quser_description))
                                visited_user_ids.add(quser_id_str)

                except EOFError:
                    self.logger.error('EOFError: {0} ended before the logical end-of-stream was detected,'.format(filename))

            tweet_output.close()
            user_output.close()
            filedata.close()
            self.logger.debug('{0} done!'.format(filename))
            print('{0} done!'.format(filename))
