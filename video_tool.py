#!/usr/bin/env python3
"""Video Transcription CLI Tool
A tool to download videos, transcribe them with Whisper, and search through transcriptions."""
import os
import sys
import sqlite3
import subprocess
import json
from pathlib import Path
import whisper
import argparse
from datetime import datetime
import re

class VideoTranscriptionTool:
    def __init__(self, db_path="video_transcriptions.db", download_dir="downloads"):
        self.db_path = db_path
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.init_database()
        self.whisper_model = None

    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                filename TEXT NOT NULL,
                url TEXT NOT NULL,
                download_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transcriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER,
                text TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos (id) ON DELETE CASCADE
            )
        ''')
        conn.commit()
        conn.close()

    def load_whisper_model(self, model_size="base"):
        """Load Whisper model"""
        if self.whisper_model is None:
            print(f"Loading Whisper model ({model_size})...")
            self.whisper_model = whisper.load_model(model_size)
            print("Whisper model loaded!")
        return self.whisper_model

    def download_video(self, url):
        """Download video using yt-dlp"""
        print(f"Downloading video from: {url}")
        # Get video info first
        try:
            info_cmd = [
                "yt-dlp",
                "--print", "%(title)s",
                "--print", "%(ext)s",
                url
            ]
            result = subprocess.run(info_cmd, capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')
            title = lines[0]
            ext = lines[1]
            # Clean title for filename
            safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)
            filename = f"{safe_title}.{ext}"
            filepath = self.download_dir / filename
            # Download the video (max 720p)
            download_cmd = [
                "yt-dlp",
                "-f", "best[height<=720]",
                "-o", str(filepath),
                url
            ]
            subprocess.run(download_cmd, check=True)
            print(f"Downloaded: {title}")
            return title, str(filepath), url
        except subprocess.CalledProcessError as e:
            print(f"Error downloading video: {e}")
            return None, None, None

    def transcribe_video(self, video_path, video_id):
        """Transcribe video using Whisper"""
        print(f"Transcribing video: {video_path}")
        model = self.load_whisper_model()
        result = model.transcribe(video_path, word_timestamps=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Insert segments with word-level timestamps
        for segment in result['segments']:
            if 'words' in segment:
                for word_info in segment['words']:
                    cursor.execute('''
                        INSERT INTO transcriptions (video_id, text, start_time, end_time)
                        VALUES (?, ?, ?, ?)
                    ''', (video_id, word_info['word'].strip(), word_info['start'], word_info['end']))
            else:
                # Fallback to segment-level if word timestamps not available
                cursor.execute('''
                    INSERT INTO transcriptions (video_id, text, start_time, end_time)
                    VALUES (?, ?, ?, ?)
                ''', (video_id, segment['text'].strip(), segment['start'], segment['end']))
        conn.commit()
        conn.close()
        print("Transcription completed and saved to database!")

    def add_video(self, url):
        """Download and transcribe a video"""
        title, filepath, url = self.download_video(url)
        if not filepath:
            return False
        # Add to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO videos (title, filename, url)
            VALUES (?, ?, ?)
        ''', (title, filepath, url))
        video_id = cursor.lastrowid
        conn.commit()
        conn.close()
        # Transcribe the video
        self.transcribe_video(filepath, video_id)
        return True

    def search_transcriptions(self, query):
        """Search for words/phrases in transcriptions"""
        if ' ' in query.strip():
            return self.phrase_search(query)
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT t.id, v.id as video_id, v.title, v.filename, t.text, t.start_time, t.end_time
                FROM transcriptions t
                JOIN videos v ON t.video_id = v.id
                WHERE LOWER(t.text) LIKE LOWER(?)
                ORDER BY v.title, t.start_time
            ''', (f'%{query}%',))
            results = cursor.fetchall()
            conn.close()
            return results

    def phrase_search(self, phrase):
        """Search for a phrase by looking for consecutive words"""
        words = phrase.lower().split()
        if len(words) == 0:
            return []

        # First word in the phrase
        first_word = words[0]

        # First, find all videos that contain the first word
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT DISTINCT v.id, v.title, v.filename
            FROM videos v
            JOIN transcriptions t ON t.video_id = v.id
            WHERE LOWER(t.text) = LOWER(?)
        ''', (first_word,))
        matching_videos = cursor.fetchall()

        results = []
        window_size = 2.0  # seconds between consecutive words in a phrase

        for video_id, title, filename in matching_videos:
            # Get all transcriptions for this video, ordered by time
            cursor.execute('''
                SELECT text, start_time, end_time
                FROM transcriptions
                WHERE video_id = ?
                ORDER BY start_time
            ''', (video_id,))
            all_words = cursor.fetchall()

            # Look for the phrase in this video's transcriptions
            for i in range(len(all_words) - len(words) + 1):
                # Check if the next len(words) words match our phrase
                match_found = True
                matched_words = []
                start_time = None
                end_time = None

                for j in range(len(words)):
                    if i + j >= len(all_words):
                        match_found = False
                        break

                    word_text, word_start, word_end = all_words[i + j]
                    if word_text.lower() != words[j].lower():
                        match_found = False
                        break

                    # Check timing between words if not the first word
                    if j > 0:
                        prev_word_end = all_words[i + j - 1][2]
                        current_word_start = word_start
                        if current_word_start - prev_word_end > window_size:
                            match_found = False
                            break

                    matched_words.append(word_text)
                    if j == 0:
                        start_time = word_start
                    if j == len(words) - 1:
                        end_time = word_end

                if match_found and start_time is not None and end_time is not None:
                    matched_text = ' '.join(matched_words)
                    results.append((
                        None,  # trans_id
                        video_id,
                        title,
                        filename,
                        matched_text,
                        start_time,
                        end_time
                    ))

        conn.close()
        return results

    def format_time(self, seconds):
        """Format seconds to MM:SS format"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def extract_clip(self, video_path, start_time, end_time, output_path):
        """Extract a clip from video using ffmpeg"""
        # Add 0.5 second buffer
        start_with_buffer = max(0, start_time - 0.5)
        end_with_buffer = end_time + 0.5
        duration = end_with_buffer - start_with_buffer
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-ss", str(start_with_buffer),
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            output_path,
            "-y"  # Overwrite output file
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error extracting clip: {e}")
            return False

    def delete_video(self, title):
        """Delete video and its transcriptions from database and filesystem"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Find video
        cursor.execute('SELECT id, filename FROM videos WHERE title = ?', (title,))
        result = cursor.fetchone()
        if not result:
            conn.close()
            return False, "Video not found in database"
        video_id, filename = result
        # Delete from database (transcriptions will be deleted due to CASCADE)
        cursor.execute('DELETE FROM videos WHERE id = ?', (video_id,))
        conn.commit()
        conn.close()
        # Delete file from filesystem
        try:
            if os.path.exists(filename):
                os.remove(filename)
                return True, f"Deleted video: {title}"
            else:
                return True, f"Deleted from database: {title} (file not found on disk)"
        except Exception as e:
            return False, f"Error deleting file: {e}"

    def list_videos(self):
        """List all videos in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT title, download_date FROM videos ORDER BY download_date DESC')
        results = cursor.fetchall()
        conn.close()
        return results

    def get_downloads_folder(self):
        """Get the user's Downloads folder, fallback to clips directory"""
        # Try to find user's Downloads folder
        home = Path.home()
        downloads_paths = [
            home / "Downloads",
            home / "downloads",
            home / "Desktop"  # Fallback for some systems
        ]
        for path in downloads_paths:
            if path.exists() and path.is_dir():
                return path
        # Fallback to clips directory in project
        clips_dir = Path("clips")
        clips_dir.mkdir(exist_ok=True)
        return clips_dir

    def search_mode(self):
        """Interactive search mode"""
        print("\n=== Search Mode ===")
        print("Enter search terms to find words/phrases in transcriptions")
        print("Commands: 'download <clip#>' | 'download full <clip#>' | 'delete <video>' | 'list' | 'back' to return")
        print("Note: You can now search for phrases by entering multiple words")
        print()
        current_search_results = []
        video_transcriptions = {}  # Store this between commands

        while True:
            try:
                user_input = input("Search >>> ").strip()
                if not user_input:
                    continue
                if user_input.lower() == 'back':
                    break
                elif user_input.lower() == 'list':
                    videos = self.list_videos()
                    if videos:
                        print("\nVideos in database:")
                        for title, date in videos:
                            print(f"  - {title} (downloaded: {date})")
                    else:
                        print("No videos in database.")
                    print()
                elif user_input.lower().startswith('download '):
                    try:
                        parts = user_input.lower().split()
                        if len(parts) >= 3:
                            if parts[1] == 'full':
                                # Format: download full <clip#>
                                if len(parts) >= 4:
                                    clip_num = int(parts[3].strip())
                                    download_full = True
                                else:
                                    print("Usage: download full <clip#>")
                                    continue
                            elif len(parts) >= 3 and parts[2] == 'full':
                                # Format: download <clip#> full
                                clip_num = int(parts[1].strip())
                                download_full = True
                            else:
                                # Original format: download <clip#>
                                clip_num = int(parts[1].strip())
                                download_full = False
                        else:
                            # Original format: download <clip#>
                            clip_num = int(parts[1].strip())
                            download_full = False

                        if not current_search_results:
                            print("No search results available. Search for something first.")
                            continue

                        if 1 <= clip_num <= len(current_search_results):
                            result = current_search_results[clip_num - 1]

                            # Handle both phrase search results and word search results
                            try:
                                if len(result) >= 7:
                                    trans_id, video_id, title, filename, text, start_time, end_time = result[:7]
                                else:
                                    trans_id, title, filename, text, start_time, end_time = result[:6]
                                    video_id = None  # Not available in original format
                            except Exception as e:
                                print(f"Error parsing result: {e}")
                                continue

                            if download_full:
                                if video_id is None:
                                    print("Cannot download full context for this result (missing video ID)")
                                    continue

                                # Get the transcriptions for this video
                                transcriptions = video_transcriptions.get(video_id, [])
                                if not transcriptions:
                                    conn = sqlite3.connect(self.db_path)
                                    cursor = conn.cursor()
                                    cursor.execute('''
                                        SELECT text, start_time, end_time
                                        FROM transcriptions
                                        WHERE video_id = ?
                                        ORDER BY start_time
                                    ''', (video_id,))
                                    transcriptions = cursor.fetchall()
                                    video_transcriptions[video_id] = transcriptions
                                    conn.close()

                                if not transcriptions:
                                    print("No transcriptions available for this video")
                                    continue

                                # Find the full sentence time range
                                window_size = 15  # seconds
                                full_start_time = start_time - window_size
                                full_end_time = end_time + window_size

                                # Find the actual start time (earliest word that starts before but overlaps with our window)
                                actual_start_time = full_start_time
                                for word_text, word_start, word_end in transcriptions:
                                    if word_end <= full_start_time:
                                        continue
                                    if word_start < full_start_time:
                                        actual_start_time = word_start
                                    break

                                # Find the actual end time (latest word that ends after but starts before our window end)
                                actual_end_time = full_end_time
                                for word_text, word_start, word_end in reversed(transcriptions):
                                    if word_start <= full_end_time:
                                        if word_end > actual_end_time:
                                            actual_end_time = word_end
                                        break

                                # Generate clip filename
                                downloads_dir = self.get_downloads_folder()
                                safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)
                                clip_filename = f"clip_{clip_num}_full_{safe_title}_{self.format_time(actual_start_time).replace(':', '-')}_to_{self.format_time(actual_end_time).replace(':', '-')}.mp4"
                                clip_path = downloads_dir / clip_filename

                                # Combine all words in the window to get the full sentence
                                sentence_parts = []
                                for word_text, word_start, word_end in transcriptions:
                                    if actual_start_time <= word_start <= actual_end_time or \
                                       actual_start_time <= word_end <= actual_end_time or \
                                       (word_start <= actual_start_time and word_end >= actual_end_time):
                                        sentence_parts.append(word_text)
                                full_sentence = ' '.join(sentence_parts)

                                print(f"Extracting full context clip {clip_num}: '{full_sentence}' from {self.format_time(actual_start_time)}-{self.format_time(actual_end_time)}")
                                print(f"Saving to: {downloads_dir}")
                                if self.extract_clip(filename, actual_start_time, actual_end_time, str(clip_path)):
                                    print(f"Clip saved as: {clip_path}")
                                else:
                                    print("Failed to extract clip.")
                            else:
                                # Original download functionality
                                downloads_dir = self.get_downloads_folder()
                                safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)
                                clip_filename = f"clip_{clip_num}_{safe_title}_{self.format_time(start_time).replace(':', '-')}.mp4"
                                clip_path = downloads_dir / clip_filename

                                print(f"Extracting clip {clip_num}: '{text}' from {self.format_time(start_time)}-{self.format_time(end_time)}")
                                print(f"Saving to: {downloads_dir}")
                                if self.extract_clip(filename, start_time, end_time, str(clip_path)):
                                    print(f"Clip saved as: {clip_path}")
                                else:
                                    print("Failed to extract clip.")
                        else:
                            print(f"Invalid clip number. Choose between 1 and {len(current_search_results)}")
                    except ValueError:
                        print("Invalid clip number format.")
                    except Exception as e:
                        print(f"Error downloading clip: {e}")
                elif user_input.lower().startswith('delete '):
                    title = user_input[7:].strip()
                    if not title:
                        print("Please provide a video title to delete.")
                        continue
                    success, message = self.delete_video(title)
                    print(message)
                else:
                    # Treat as search query
                    query = user_input
                    print(f"Searching for: '{query}'")
                    results = self.search_transcriptions(query)
                    current_search_results = results
                    video_transcriptions = {}  # Reset for new search

                    if results:
                        print(f"\nFound {len(results)} matches:")
                        # Get all unique video_ids from results
                        video_ids = set()
                        for result in results:
                            if len(result) > 1:  # Make sure we have enough elements
                                video_ids.add(result[1])  # video_id is at index 1

                        # Create a dictionary to hold all transcriptions for these videos
                        if video_ids:
                            conn = sqlite3.connect(self.db_path)
                            cursor = conn.cursor()

                            for video_id in video_ids:
                                cursor.execute('''
                                    SELECT text, start_time, end_time
                                    FROM transcriptions
                                    WHERE video_id = ?
                                    ORDER BY start_time
                                ''', (video_id,))
                                video_transcriptions[video_id] = cursor.fetchall()

                            conn.close()

                        # Now display the results with context
                        window_size = 15  # seconds before and after

                        for i, result in enumerate(results, 1):
                            try:
                                if len(result) == 7:  # Phrase search results have 7 elements
                                    trans_id, video_id, title, filename, text, start_time, end_time = result
                                else:  # Original word search results might have a different format
                                    # Handle cases where result might not have all elements
                                    if len(result) >= 7:
                                        trans_id, video_id, title, filename, text, start_time, end_time = result[:7]
                                    else:
                                        # Skip or handle malformed results
                                        continue

                                # Get transcriptions for this video (if available)
                                transcriptions = video_transcriptions.get(video_id, [])

                                # Only try to get context if we have transcriptions
                                if transcriptions:
                                    # Find all words within window_size seconds before and after
                                    window_start = start_time - window_size
                                    window_end = end_time + window_size

                                    # Combine text of all words in the window
                                    sentence_parts = []
                                    for word_text, word_start, word_end in transcriptions:
                                        if (window_start <= word_start <= window_end or
                                            window_start <= word_end <= window_end or
                                            (word_start <= window_start and word_end >= window_end)):
                                            sentence_parts.append(word_text)

                                    # Join the words to form the sentence
                                    full_sentence = ' '.join(sentence_parts)
                                else:
                                    full_sentence = "Context not available"

                                # Print the original result with the full sentence
                                print(f"  {i}. [{self.format_time(start_time)}-{self.format_time(end_time)}] "
                                      f"'{text}' in '{title}'")
                                print(f"     Context: {full_sentence}\n")  # Added newline for spacing
                            except Exception as e:
                                print(f"Error processing result {i}: {e}")
                                continue
                    else:
                        print("No matches found.")
                    print()
            except KeyboardInterrupt:
                print("\nReturning to main menu...")
                break
            except Exception as e:
                print(f"Error: {e}")

    def run_cli(self):
        """Main CLI loop"""
        print("=== Video Transcription Tool ===")
        print("Commands:")
        print("  - Paste a video URL to download and transcribe")
        print("  - 'search' to enter search mode")
        print("  - 'list' to show all videos")
        print("  - 'quit' to exit")
        print()
        while True:
            try:
                user_input = input(">>> ").strip()
                if not user_input:
                    continue
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'list':
                    videos = self.list_videos()
                    if videos:
                        print("\nVideos in database:")
                        for title, date in videos:
                            print(f"  - {title} (downloaded: {date})")
                    else:
                        print("No videos in database.")
                    print()
                elif user_input.lower() == 'search':
                    self.search_mode()
                elif user_input.startswith(
                        ('http://', 'https://', 'www.')) or 'youtube.com' in user_input or 'youtu.be' in user_input:
                    print("Processing video URL...")
                    if self.add_video(user_input):
                        print("Video downloaded and transcribed successfully!")
                        print("Entering search mode...")
                        self.search_mode()
                    else:
                        print("Failed to process video.")
                    print()
                else:
                    print("Unknown command. Try:")
                    print("  - Paste a video URL")
                    print("  - 'search' to search transcriptions")
                    print("  - 'list' to show all videos")
                    print("  - 'quit' to exit")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Video Transcription CLI Tool')
    parser.add_argument('--db', default='video_transcriptions.db', help='Database file path')
    parser.add_argument('--downloads', default='downloads', help='Download directory')
    parser.add_argument('--whisper-model', default='base', choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size')
    args = parser.parse_args()

    # Check dependencies
    missing = []
    # Check yt-dlp
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append('yt-dlp')
    # Check ffmpeg with multiple possible commands
    ffmpeg_found = False
    for cmd in ['ffmpeg -version', 'ffmpeg --version', 'ffmpeg -h']:
        try:
            subprocess.run(cmd.split(), capture_output=True, check=True)
            ffmpeg_found = True
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    if not ffmpeg_found:
        missing.append('ffmpeg')
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        if 'yt-dlp' in missing:
            print("  pip install yt-dlp")
        if 'ffmpeg' in missing:
            print("  # Install ffmpeg from https://ffmpeg.org/")
        print("\nNote: If you have ffmpeg installed but it's still showing as missing,")
        print("make sure it's in your system PATH or try running the script anyway.")
        response = input("Continue anyway? (y/n): ").lower().strip()
        if response != 'y':
            sys.exit(1)
    try:
        import whisper
    except ImportError:
        print("OpenAI Whisper not found. Please install it:")
        print("  pip install openai-whisper")
        sys.exit(1)

    tool = VideoTranscriptionTool(args.db, args.downloads)
    tool.run_cli()

if __name__ == "__main__":
    main()
