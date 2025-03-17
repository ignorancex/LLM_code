import json
import os
from datetime import datetime
from pathlib import Path
from re import sub
from typing import Tuple

import googleapiclient.discovery
import pandas as pd

RAW_DATA_PATH = "raw_jsons"


def init_service():
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    api_service_name = "youtube"
    api_version = "v3"
    API_KEY = open(".API_KEY").read().strip()
    print(API_KEY)
    return googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=API_KEY
    )


def get_channel_info(api_service, channel_id, dump_file=False):
    channel_info = (
        api_service.channels()
        .list(
            id=channel_id,
            part="brandingSettings,contentDetails,snippet,statistics,status,topicDetails",
        )
        .execute()
    )
    if dump_file:
        path = f"{RAW_DATA_PATH}/{channel_id}/channel_info/"
        Path(path).mkdir(parents=True, exist_ok=True)
        json.dump(channel_info, open(f"{path}/{channel_id}_channel_info.json", "w"))
    return channel_info


def get_channel_videos(
    api_service, playlist_id, channel_id, max_videos=10, dump_file=False
):
    videos = []
    next_page_token = None
    page = 0
    n_videos = 0
    while True:
        response = (
            api_service.playlistItems()
            .list(
                playlistId=playlist_id,
                part="contentDetails,snippet",
                maxResults=50,
                pageToken=next_page_token,
            )
            .execute()
        )
        if dump_file:
            path = f"{RAW_DATA_PATH}/{channel_id}/videos/"
            Path(path).mkdir(parents=True, exist_ok=True)
            json.dump(response, open(f"{path}/{playlist_id}_video_{page}.json", "w"))
        videos.extend(response["items"])
        next_page_token = response.get("nextPageToken")
        n_videos += 50
        page += 1
        if next_page_token is None or n_videos >= max_videos:
            break
    return videos


def get_video_data(api_service, video_id, channel_id, dump_file=False):
    video_data = (
        api_service.videos()
        .list(id=video_id, part="contentDetails,snippet,statistics,status,topicDetails")
        .execute()
    )
    if dump_file:
        path = f"{RAW_DATA_PATH}/{channel_id}/video_data/"
        Path(path).mkdir(parents=True, exist_ok=True)
        json.dump(video_data, open(f"{path}/{video_id}_data.json", "w"))
    return video_data


def get_video_stats(api_service, video_id, channel_id, dump_file=False):
    video_stats = api_service.videos().list(id=video_id, part="statistics").execute()
    if dump_file:
        path = f"{RAW_DATA_PATH}/{channel_id}/video_stats/"
        Path(path).mkdir(parents=True, exist_ok=True)
        json.dump(video_stats, open(f"{path}/{video_id}_stats.json", "w"))
    return video_stats


def get_video_comments(
    api_service, video_id, channel_id, max_comments=10000, dump_file=False
):
    comments = []
    next_page_token = None
    page = 0
    n_comments = 0
    while True:
        response = (
            api_service.commentThreads()
            .list(
                videoId=video_id,
                part="snippet",
                maxResults=100,
                textFormat="plainText",
                pageToken=next_page_token,
            )
            .execute()
        )
        if dump_file:
            path = f"{RAW_DATA_PATH}/{channel_id}/comments/"
            Path(path).mkdir(parents=True, exist_ok=True)
            json.dump(response, open(f"{path}/{video_id}_comments_{page}.json", "w"))
        comments.extend(response["items"])
        next_page_token = response.get("nextPageToken")
        n_comments += 100
        page += 1
        if next_page_token is None or n_comments >= max_comments:
            break
    return comments


# channel_id1,channel_id2,...
def retrieve_full_channel_data(
    api_service,
    channel_ids,
    max_videos=10,
    max_comments=10000,
    video_date_range=None,
    dump_file=False,
):
    channels_dataset = []
    videos_dataset = []
    comments_dataset = []
    # Retrieving the channels
    channel_info = get_channel_info(api_service, channel_ids, dump_file=dump_file)
    for channel in channel_info["items"]:
        channel_id = channel["id"]
        print(channel_id)
        # Filtering the response to get the relevant info
        channels_dataset.append(filter_channel_info(channel))
        # Retrieving the videos. We need the playlist id first!
        playlist_id = channel["contentDetails"]["relatedPlaylists"]["uploads"]
        videos = get_channel_videos(
            api_service,
            playlist_id,
            channel_id=channel_id,
            max_videos=max_videos,
            dump_file=dump_file,
        )
        # Retrieving video stats + comments
        for video in videos:
            video_id = video["contentDetails"]["videoId"]
            # print(f"video_id: {video_id}")
            video_published_date = video["snippet"]["publishedAt"]
            if not video_date_range or _is_within_date_range(
                video_published_date, video_date_range
            ):
                video_data = get_video_data(
                    api_service, video_id, channel_id=channel_id, dump_file=dump_file
                )
                videos_dataset.append(filter_video_info(video_data["items"][0]))
                if max_comments > 0 and (
                    "commentCount" in video_data["items"][0].get("statistics")
                    and int(
                        video_data["items"][0].get("statistics").get("commentCount")
                    )
                    > 0
                ):
                    comments = get_video_comments(
                        api_service,
                        video_id,
                        channel_id=channel_id,
                        max_comments=max_comments,
                        dump_file=dump_file,
                    )
                    for comment in comments:
                        filtered_comment_info = filter_comment_info(comment)
                        filtered_comment_info.extend([video_id, channel_id])
                        comments_dataset.append(filtered_comment_info)
                else:
                    comments_dataset.append(["comments disabled", video_id, channel_id])
    return (channels_dataset, videos_dataset, comments_dataset)


def retrieve_channel_id(
    api_service, media_author_name: str, media_author_url: str
) -> str:
    """
    Searches for a channel ID using the author name and URL extracted from a Reddit thread.

    Args:
        media_author_name: The name of the channel.
        media_author_url: The URL of the channel.

    Returns:
        The channel ID, or "None" if no channel was found.
    """
    channel_id = "None"
    response = (
        api_service.channels().list(part="id", forUsername=media_author_name).execute()
    )
    if response.get("items"):
        channel_id = response["items"][0]["id"]
    else:
        response = (
            api_service.search()
            .list(part="snippet", q=media_author_url, type="video", maxResults=1)
            .execute()
        )
        if response.get("items"):
            channel_title = response["items"][0]["snippet"]["channelTitle"]
            channel_id = response["items"][0]["snippet"]["channelId"]
            if channel_title != media_author_name:
                print(
                    f"WARNING: channel name '{channel_title}' does not match media author name '{media_author_name}'"
                )
                channel_id = f"<UNMATCHED>{channel_id}"
    return channel_id


def filter_channel_info(response_item):
    return {
        "id": response_item.get("id"),
        "title": response_item.get("snippet").get("title"),
        "description": response_item.get("snippet").get("description"),
        "country": response_item.get("snippet").get("country"),
        "published_at": response_item.get("snippet").get("publishedAt"),
        "view_count": response_item.get("statistics").get("viewCount"),
        "sub_count": response_item.get("statistics").get("subscriberCount"),
        "video_count": response_item.get("statistics").get("videoCount"),
        "topic_categories": ",".join(
            response_item.get("topicDetails").get("topicCategories")
        )
        if response_item.get("topicDetails")
        else None,
        "made_for_kids": response_item.get("status").get("madeForKids"),
        "keywords": response_item.get("brandingSettings")
        .get("channel")
        .get("keywords"),
    }


def filter_video_info(video_response_item):
    return {
        "video_id": video_response_item.get("id"),
        "title": video_response_item.get("snippet").get("title"),
        "description": video_response_item.get("snippet").get("description"),
        "tags": ",".join(video_response_item.get("snippet").get("tags"))
        if video_response_item.get("snippet").get("tags")
        else None,
        "category_id": video_response_item.get("snippet").get("categoryId"),
        "published_at": video_response_item.get("snippet").get("publishedAt"),
        "duration": video_response_item.get("contentDetails").get("duration"),
        "made_for_kids": video_response_item.get("status").get("madeForKids"),
        "channel": video_response_item.get("snippet").get("channelTitle"),
        "view_count": video_response_item.get("statistics").get("viewCount"),
        "like_count": video_response_item.get("statistics").get("likeCount"),
        "dislike_count": video_response_item.get("statistics").get("dislikeCount"),
        "favourite_count": video_response_item.get("statistics").get("favoriteCount"),
        "comment_count": video_response_item.get("statistics").get("commentCount"),
        "topic_categories": ",".join(
            video_response_item.get("topicDetails").get("topicCategories")
        )
        if video_response_item.get("topicDetails")
        else None,
    }


def filter_comment_info(comment_response_item):
    id = comment_response_item.get("id")
    text = (
        comment_response_item.get("snippet")
        .get("topLevelComment")
        .get("snippet")
        .get("textOriginal")
    )
    author_name = (
        comment_response_item.get("snippet")
        .get("topLevelComment")
        .get("snippet")
        .get("authorDisplayName")
    )
    author_id = (
        (
            comment_response_item.get("snippet")
            .get("topLevelComment")
            .get("snippet")
            .get("authorChannelId")
            .get("value")
        )
        if "authorChannelId"
        in comment_response_item.get("snippet").get("topLevelComment").get("snippet")
        else ""
    )
    like_count = (
        comment_response_item.get("snippet")
        .get("topLevelComment")
        .get("snippet")
        .get("likeCount")
    )
    published_at = (
        comment_response_item.get("snippet")
        .get("topLevelComment")
        .get("snippet")
        .get("publishedAt")
    )
    updated_at = (
        comment_response_item.get("snippet")
        .get("topLevelComment")
        .get("snippet")
        .get("updatedAt")
    )
    reply_count = comment_response_item.get("snippet").get("totalReplyCount")
    return [
        id,
        text,
        author_name,
        author_id,
        like_count,
        published_at,
        updated_at,
        reply_count,
    ]


def create_channels_df(channels_dataset):
    return pd.DataFrame(channels_dataset)


def create_videos_df(videos_dataset):
    return pd.DataFrame(videos_dataset)


def create_comments_df(comments_dataset):
    return pd.DataFrame(
        comments_dataset,
        columns=[
            "id",
            "text",
            "author_name",
            "author_id",
            "like_count",
            "published_at",
            "updated_at",
            "reply_count",
            "video_id",
            "channel_id",
        ],
    )


def _parse_date_range(date_range: str) -> Tuple[datetime.date, datetime.date]:
    """
    Parses a date range string into a tuple of datetime.date objects.

    Args:
        date_range: A string of the form "DD/MM/YYYY-DD/MM/YYYY"

    Returns:
        A tuple of datetime.date objects (start, end).
    """
    start_date, end_date = date_range.split("-")
    return (
        datetime.strptime(start_date, "%d/%m/%Y").date(),
        datetime.strptime(end_date, "%d/%m/%Y").date(),
    )


def _is_within_date_range(date: str, date_range: str) -> bool:
    """
    Checks if a date is within a date range.

    Args:
        date: A string in ISO 8601 format.
        date_range: A string of the form "DD/MM/YYYY-DD/MM/YYYY"

    Returns:
        True if the date is within the date range, False otherwise.
    """
    start_date, end_date = _parse_date_range(date_range)
    return (
        start_date <= datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ").date() <= end_date
    )
