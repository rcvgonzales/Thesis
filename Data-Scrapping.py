#!/usr/bin/env python3
"""
pilot_scraper.py
- Input: pilot_input.csv (ClaimID,URL)
- Output: /ThesisData/<ClaimID>/<videofile>, <audiofile>, <transcript.txt>
- Updates master_metadata.csv with one row per video.
"""

import os
import csv
import json
import math
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from yt_dlp import YoutubeDL
from moviepy import VideoFileClip

# Optional ASR (Whisper)
try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

# CONFIG
BASE_DIR = Path.cwd() / "ThesisData"
INPUT_CSV = Path.cwd() / "pilot_input.csv"
MASTER_META = Path.cwd() / "master_metadata.csv"
SUPERSPREADER_THRESHOLD = 100_000  # followers/subscribers threshold

# ensure base dir exists
BASE_DIR.mkdir(parents=True, exist_ok=True)

# yt-dlp options: keep the original file name unique, and write json metadata
YTDL_OPTS = {
    "outtmpl": str(BASE_DIR / "%(id)s.%(ext)s"),
    "writesubtitles": False,
    "writeinfojson": True,
    "quiet": True,
    "format": "bestvideo+bestaudio/best",
    # add any cookies or headers if needed e.g. for region-locked content (not recommended)
}

ytdl = YoutubeDL(YTDL_OPTS)

def download_video(url):
    """
    Downloads video using yt-dlp library and returns the info dict and paths.
    """
    try:
        info = ytdl.extract_info(url, download=True)
        # info contains 'id', 'ext', 'title', 'uploader', 'upload_date', 'view_count' etc.
        video_filename = ytdl.prepare_filename(info)
        # info json: yt-dlp also writes <id>.info.json in same folder if writeinfojson True.
        infojson_path = Path(video_filename).with_suffix(".info.json")
        return info, Path(video_filename), infojson_path
    except Exception as e:
        print(f"[ERROR] failed to download {url}: {e}")
        return None, None, None

def extract_audio(video_path, target_wav_path):
    """
    Extract a single-channel WAV audio using moviepy/ffmpeg.
    """
    try:
        clip = VideoFileClip(str(video_path))
        # write audio as wav
        clip.audio.write_audiofile(str(target_wav_path), fps=16000, nbytes=2, codec="pcm_s16le")
        clip.close()
        return True
    except Exception as e:
        print(f"[ERROR] extracting audio from {video_path}: {e}")
        return False

def transcribe_whisper(audio_path, model="small"):
    if not WHISPER_AVAILABLE:
        return None
    try:
        model = whisper.load_model(model)
        # return (text, raw_result)
        res = model.transcribe(str(audio_path))
        return res.get("text", None), res
    except Exception as e:
        print(f"[WARN] Whisper transcription failed: {e}")
        return None

def parse_upload_date(info):
    # yt-dlp upload_date format 'YYYYMMDD' or missing
    ud = info.get("upload_date")
    if ud:
        try:
            return datetime.strptime(ud, "%Y%m%d").date().isoformat()
        except:
            return ud
    return None

def guess_superspreader(info):
    # yt-dlp metadata has 'channel_follower_count' sometimes; fallback to view_count heuristic
    subs = info.get("channel_follower_count") or info.get("subscriber_count") or None
    if subs:
        try:
            return int(subs) >= SUPERSPREADER_THRESHOLD
        except:
            pass
    # fallback: if view_count > threshold -> treat as superspreader content
    views = info.get("view_count") or 0
    try:
        return int(views) >= SUPERSPREADER_THRESHOLD
    except:
        return False

def metadata_row_from_info(info, claim_id, video_path, audio_path, transcript_path, infojson_path):
    return {
        "Claim ID": claim_id,
        "Video ID": info.get("id"),
        "URL": info.get("webpage_url"),
        "Platform": "TikTok" if "tiktok" in info.get("webpage_url", "") else "YouTube",
        "Channel/Account": info.get("uploader") or info.get("channel") or "",
        "Is_Superspreader": guess_superspreader(info),
        "Upload Date": parse_upload_date(info),
        "Views/Likes": f"views:{info.get('view_count')} likes:{info.get('like_count')}",
        "Claim Category": "",  # you can map ClaimID->Category separately
        "Annotation Status": "Raw Downloaded",
        "Transcription File": str(transcript_path) if transcript_path else "",
        "Audio File": str(audio_path) if audio_path else "",
        "InfoJSON": str(infojson_path) if infojson_path else "",
        "Title": info.get("title", ""),
        "Uploader_Id": info.get("uploader_id", ""),
    }

def append_row_to_master(row):
    df = pd.DataFrame([row])
    if MASTER_META.exists():
        df.to_csv(MASTER_META, mode="a", header=False, index=False)
    else:
        df.to_csv(MASTER_META, index=False)

def main():
    # load input CSV
    inputs = pd.read_csv(INPUT_CSV)
    # if master exists and you want a fresh one, comment out the following
    if not MASTER_META.exists():
        # create header
        pd.DataFrame(columns=[
            "Claim ID","Video ID","URL","Platform","Channel/Account","Is_Superspreader",
            "Upload Date","Views/Likes","Claim Category","Annotation Status",
            "Transcription File","Audio File","InfoJSON","Title","Uploader_Id"
        ]).to_csv(MASTER_META, index=False)

    # optional whisper model
    whisper_model = "small" if WHISPER_AVAILABLE else None

    for _, row in tqdm(inputs.iterrows(), total=len(inputs)):
        claim = str(row["ClaimID"])
        url = str(row["URL"])
        claim_folder = BASE_DIR / claim
        claim_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n[INFO] Downloading {url} for claim {claim} ...")
        info, video_path, infojson_path = download_video(url)
        if info is None:
            continue

        # move video into claim folder (yt-dlp wrote to BASE_DIR/<id>.<ext>)
        if video_path.exists():
            dest_video = claim_folder / video_path.name
            video_path.rename(dest_video)
        else:
            dest_video = None

        # move infojson if exists
        if infojson_path and infojson_path.exists():
            dest_infojson = claim_folder / infojson_path.name
            infojson_path.rename(dest_infojson)
        else:
            dest_infojson = None

        # extract audio
        audio_fname = f"{info.get('id')}.wav"
        audio_path = claim_folder / audio_fname
        audio_ok = False
        if dest_video:
            audio_ok = extract_audio(dest_video, audio_path)
        if not audio_ok:
            audio_path = ""

        # whisper transcription (optional)
        transcript_path = ""
        if WHISPER_AVAILABLE and audio_ok:
            print(f"[INFO] Transcribing {audio_path.name} with Whisper ({whisper_model}) ...")
            text, raw_res = transcribe_whisper(audio_path, model=whisper_model)
            if text:
                transcript_path = claim_folder / f"{info.get('id')}.txt"
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(text)

        # create metadata row and append to master
        mrow = metadata_row_from_info(info, claim, dest_video, audio_path, transcript_path, dest_infojson)
        append_row_to_master(mrow)
        print(f"[OK] Saved files to {claim_folder}. Metadata appended to {MASTER_META}.")

if __name__ == "__main__":
    main()
