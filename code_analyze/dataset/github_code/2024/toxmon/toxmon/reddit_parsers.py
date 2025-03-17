import ast
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RedditParser:
    url_pattern: re.Pattern = field(
        default=re.compile(
            r"(https?:\/\/(?:www\.)?[\w\-]+(?:\.[\w\-]+)+[\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])"
        ),
        init=False,
    )

    def parse_thread(self, thread_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts essential information from the thread JSON and returns a Pandas DataFrame.
        """
        thread_name: Optional[str] = thread_json.get("name", None)
        subreddit_name: Optional[str] = thread_json.get("subreddit_name_prefixed", None)
        subreddit_id: Optional[str] = thread_json.get("subreddit", {}).get("id", None)
        title: Optional[str] = thread_json.get("title", None)
        author_name: Optional[str] = (
            thread_json.get("author", {}).get("name", None)
            if thread_json.get("author") != "None"
            else None
        )
        num_comments: int = thread_json.get("num_comments", 0)
        ups: int = thread_json.get("ups", 0)
        downs: int = thread_json.get("downs", 0)
        upvote_ratio: float = thread_json.get("upvote_ratio", 0.0)
        score: int = thread_json.get("score", 0)
        selftext: Optional[str] = thread_json.get("selftext", None)
        media: Optional[str] = thread_json.get("media", None)
        media_only: bool = thread_json.get("media_only", False)
        created_utc: int = thread_json.get("created_utc", 0)

        thread_dict = {
            "thread_name": thread_name,
            "subreddit_name": subreddit_name,
            "subreddit_id": subreddit_id,
            "title": title,
            "author_name": author_name,
            "num_comments": num_comments,
            "ups": ups,
            "downs": downs,
            "upvote_ratio": upvote_ratio,
            "score": score,
            "selftext": selftext,
            "media": media,
            "media_only": media_only,
            "created_utc": created_utc,
            "urls": self.extract_urls_from_selftext(thread_json),
        }
        # Add the keys from the YouTube media dictionary to the thread dictionary
        yt_data = self.extract_youtube_media(thread_dict)
        if yt_data != "None":
            thread_dict.update(yt_data)

        return thread_dict

    def parse_comment(self, comment_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts essential information from the comment JSON and returns a Pandas DataFrame.
        """
        return self._parse_comment_tree(comment_json)

    def _parse_comment_tree(
        self, comment: Dict[str, Any], parent_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        Recursively parses the comment tree to extract information from both top-level comments and nested replies.
        """
        comment_data = [
            {
                "id": comment.get("id", None),
                "body": comment.get("body", None),
                "author_name": comment.get("author", None),
                "created_utc": comment.get("created_utc", None),
                "score": comment.get("score", 0),
                "is_submitter": comment.get("is_submitter", False),
                "parent_id": parent_id,
                "link_id": comment.get("link_id", None),
                "permalink": comment.get("permalink", None),
                "controversiality": comment.get("controversiality", 0),
                "gilded": comment.get("gilded", 0),
            }
        ]

        replies = comment.get("replies", [])
        for reply in replies:
            comment_data.extend(
                self._parse_comment_tree(reply, parent_id=comment.get("id"))
            )

        return comment_data

    def extract_youtube_media(self, parsed_thread: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts YouTube media information from the provided data.
        """
        media_str = parsed_thread.get("media")
        if media_str == "None":
            return "None"
        else:
            try:
                media_dict = ast.literal_eval(media_str)
                if "type" in media_dict and media_dict["type"] == "youtube.com":
                    youtube_data = {
                        "media_author_name": media_dict["oembed"].get(
                            "author_name", "None"
                        ),
                        "media_author_url": media_dict["oembed"].get(
                            "author_url", "None"
                        ),
                        "video_title": media_dict["oembed"].get("title", "None"),
                        "video_id": self._extract_video_id(
                            media_dict["oembed"]["html"]
                        ),
                    }
                    return youtube_data
            except ValueError:
                print(f"Error parsing media: {media_str}")
        return "None"

    def extract_urls_from_selftext(self, parsed_data: Dict[str, Any]) -> List[str]:
        """
        Extracts URLs from the 'selftext' field of the provided data.
        """
        urls = self.url_pattern.findall(parsed_data.get("selftext"))
        return urls

    def _extract_video_id(self, html: str) -> str:
        """
        Extracts the YouTube video ID from the embed HTML.
        """
        video_id_match = re.search(r"youtube\.com/embed/([^/?]+)", html)
        return video_id_match.group(1) if video_id_match else "None"
