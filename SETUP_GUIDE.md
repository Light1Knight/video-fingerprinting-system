# Smart Video Fingerprinting System - Setup Guide

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
python smart_fingerprint_api.py
```
The server will start at `http://localhost:5000`

### 3. Test the System
```bash
python smart_api_client.py
```

### 4. Analyze Your Videos
```bash
# Single video
python smart_integration.py "path/to/video.mp4"

# Entire directory
python smart_integration.py "path/to/videos/" --batch
```

## 📊 API Usage Examples

### Smart Search
```bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "fingerprint_code": "ABCD1234EFGH5678IJKL9012MNOP3456-TEST",
    "filename": "video.mp4"
  }'
```

### Add Verified Fingerprint
```bash
curl -X POST http://localhost:5000/api/fingerprints/add-verified \
  -H "Content-Type: application/json" \
  -d '{
    "fingerprint_code": "NEW123456789VIDEO987654321-VER",
    "filename": "movie.mp4",
    "title": "Movie Title",
    "media_type": "movie",
    "duration": 7200
  }'
```

### Get Statistics
```bash
curl http://localhost:5000/api/stats
```

## 🔧 Core Components

- **smart_fingerprint_api.py** - Main API server
- **smart_api_client.py** - Client library and examples
- **smart_integration.py** - Local video analysis integration
- **video_fingerprinter.py** - Core fingerprinting engine
- **reset_database.py** - Database management

## 📚 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/search` | Smart fingerprint search with multiple strategies |
| POST | `/api/fingerprints` | Store new fingerprint |
| POST | `/api/fingerprints/add-verified` | Add verified fingerprint with conflict detection |
| POST | `/api/fingerprints/add-verified/confirm` | Resolve conflicts when adding verified fingerprints |
| PUT | `/api/fingerprints/<code>/verify` | Mark fingerprint as verified |
| GET | `/api/fingerprints/unverified` | List unverified fingerprints |
| GET | `/api/stats` | Database statistics |

## 🎯 Features

- **Smart Search**: 4-step search algorithm with confidence scoring
- **Conflict Resolution**: Automatic detection of potential duplicates
- **Two-Tier System**: Verified vs unverified fingerprints
- **Automatic Cleanup**: Removes duplicate unverified entries
- **Visual Fingerprinting**: 10-minute segments with 4x6 grid analysis
- **Batch Processing**: Analyze entire video directories

This system provides everything needed for video fingerprinting and duplicate detection in a clean, focused package.
