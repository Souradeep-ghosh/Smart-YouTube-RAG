import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from langchain_core.documents import Document


class TranscriptLoader:
    """Handles YouTube transcript extraction and metadata parsing."""

    def extract_video_id(self, url: str) -> str:
        """
        Extracts the video ID from a YouTube URL.
        Supports formats:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://youtube.com/shorts/VIDEO_ID
        """
        patterns = [
            r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
            r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
            r"(?:shorts\/)([0-9A-Za-z_-]{11})",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        raise ValueError(f"Could not extract video ID from URL: {url}")

    def get_video_title(self, video_id: str) -> str:
        """
        Attempts to get the video title using pytube.
        Falls back to video ID if unavailable.
        """
        try:
            from pytube import YouTube
            yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
            return yt.title
        except Exception:
            return f"YouTube Video ({video_id})"

    def load_transcript(self, url: str):
        """
        Loads transcript from a YouTube URL and returns
        a list of LangChain Documents with metadata.
        """
        video_id = self.extract_video_id(url)
        title = self.get_video_title(video_id)

        try:
            ytt_api = YouTubeTranscriptApi()
            fetched = ytt_api.fetch(video_id)
            transcript_list = [
                {
                    "text": s.text,
                    "start": s.start,
                    "duration": s.duration
                }
                for s in fetched
            ]
        except TranscriptsDisabled:
            raise ValueError("Transcripts are disabled for this video.")
        except NoTranscriptFound:
            raise ValueError("No transcript found for this video.")
        except Exception as e:
            raise ValueError(f"Error fetching transcript: {e}")

        # Convert each transcript segment into a LangChain Document
        documents = []
        for segment in transcript_list:
            documents.append(Document(
                page_content=segment["text"],
                metadata={
                    "video_id": video_id,
                    "video_title": title,
                    "start": segment["start"],
                    "duration": segment.get("duration", 0),
                    "source": url,
                }
            ))

        return documents, video_id, title