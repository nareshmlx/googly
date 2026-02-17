"""
Ensemble Data TikTok Search Tool for Agno Agent.

This tool enables searching TikTok videos by hashtag using the Ensemble Data API.
It also stores video data for display in the Discover tab.
"""

import os
from datetime import datetime, timezone

from ensembledata.api import EDClient

# Global storage for discovered videos (will be accessed by Streamlit)
_discovered_videos = []


def _format_upload_date(video: dict) -> str:
    """Extract and format upload date from API payload."""
    timestamp = (
        video.get("create_time")
        or video.get("createTime")
        or video.get("create_timestamp")
    )
    if timestamp is None:
        return "Unknown"

    try:
        ts = int(timestamp)
        # Handle potential millisecond timestamps.
        if ts > 9999999999:
            ts = ts // 1000
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return "Unknown"


def get_discovered_videos():
    """Return the list of discovered videos."""
    return _discovered_videos


def clear_discovered_videos():
    """Clear the discovered videos list."""
    global _discovered_videos
    _discovered_videos = []


def search_tiktok_hashtag(hashtag: str) -> str:
    """
    Search TikTok for videos related to a specific hashtag.

    Args:
        hashtag: The hashtag to search for (without the # symbol)

    Returns:
        A formatted string containing search results with video information
    """
    global _discovered_videos

    try:
        token = os.getenv("ENSEMBLE_API_TOKEN")
        if not token:
            return "Error: ENSEMBLE_API_TOKEN not found in environment variables."

        client = EDClient(token=token)
        result = client.tiktok.hashtag_search(hashtag=hashtag)

        if not result.data or "data" not in result.data:
            return f"No TikTok videos found for hashtag: #{hashtag}"

        videos = result.data["data"]
        if not videos:
            return f"No TikTok videos found for hashtag: #{hashtag}"

        formatted_results = [f"TikTok Search Results for #{hashtag}\n"]
        formatted_results.append(f"Found {len(videos)} videos (showing top 5)\n")
        formatted_results.append(f"API Units Charged: {result.units_charged}\n")

        for i, video in enumerate(videos[:5], 1):
            author = video.get("author", {})
            stats = video.get("statistics", {})
            video_data = video.get("video", {})
            upload_date = _format_upload_date(video)

            play_addr = video_data.get("play_addr", {})
            cover = video_data.get("cover", {})

            video_url = None
            cover_url = None

            if play_addr.get("url_list"):
                video_url = play_addr["url_list"][0]
            if cover.get("url_list"):
                cover_url = cover["url_list"][0]

            discovered_video = {
                "hashtag": hashtag,
                "video_id": video.get("aweme_id", ""),
                "author_name": author.get("nickname", "Unknown"),
                "author_username": author.get("unique_id", "unknown"),
                "description": video.get("desc", "No description")[:200],
                "likes": stats.get("digg_count", 0),
                "comments": stats.get("comment_count", 0),
                "shares": stats.get("share_count", 0),
                "views": stats.get("play_count", 0),
                "video_url": video_url,
                "cover_url": cover_url,
                "duration": video_data.get("duration", 0),
                "upload_date": upload_date,
            }

            if not any(v["video_id"] == discovered_video["video_id"] for v in _discovered_videos):
                _discovered_videos.append(discovered_video)

            formatted_results.append(f"\nVideo {i}:")
            formatted_results.append(
                f"  Author: {author.get('nickname', 'Unknown')} (@{author.get('unique_id', 'unknown')})"
            )
            formatted_results.append(
                f"  Description: {video.get('desc', 'No description')[:150]}..."
            )
            formatted_results.append(f"  Likes: {stats.get('digg_count', 0):,}")
            formatted_results.append(f"  Comments: {stats.get('comment_count', 0):,}")
            formatted_results.append(f"  Shares: {stats.get('share_count', 0):,}")
            formatted_results.append(f"  Views: {stats.get('play_count', 0):,}")
            formatted_results.append(f"  Upload Date: {upload_date}")
            formatted_results.append(f"  Video ID: {video.get('aweme_id', 'N/A')}")

        formatted_results.append(f"\n\n{min(5, len(videos))} videos added to Discover page!")
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error searching TikTok: {str(e)}"
