import os
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi

DATA_YOUTUBE_FOLDER = "data/youtube"

def ensure_folder_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        
def extract_video_id(url):
    parsed_url = urlparse(url)
    
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        query_params = parse_qs(parsed_url.query)
        return query_params.get("v", [None])[0]
    
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")
    
    return None

def get_transcript(video_id):
        transcript_items = YouTubeTranscriptApi.get_transcript(video_id)
        
        transcript_text = ""
        
        for item in transcript_items:
            transcript_text += item["text"] + "\n"
            
        return transcript_text
    
def save_transcript(video_id, url, transcript):
        ensure_folder_exists(DATA_YOUTUBE_FOLDER)
        
        filename = f"youtube_{video_id}.txt"
        file_path = os.path.join(DATA_YOUTUBE_FOLDER, filename)
        
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(f"Video ID: {video_id}\n")
            file.write(f"URL: {url}\n\n")
            file.write("Source Type: YouTube Video Transcript\n\n")
            file.write(transcript)
            
            return file_path
        
def main():
            url = input("Enter YouTube video URL: ").strip()
            video_id = extract_video_id(url)
            
            if not video_id:
                print("Invalid YouTube URL. Please try again.")
                return
            
            print("Fetching transcript... This may take a moment.")
            
            transcript = get_transcript(video_id)
            
            file_path = save_transcript(video_id, url, transcript)
            
            print( f"Transcript saved to: {file_path}")
            
            
if __name__ == "__main__":
                main()