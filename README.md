# Smart Video Fingerprinting System

A Python-based video fingerprinting system that creates unique visual fingerprints for videos and provides smart duplicate detection with an online API database.

## 🎯 Features

- **Custom Visual Fingerprinting**: Creates unique fingerprints using 10-minute segments with 4x6 grid brightness analysis
- **Smart Duplicate Detection**: Sequential and fragmented matching algorithms
- **Online API Database**: REST API for storing and querying fingerprints
- **Two-Tier System**: Verified (known videos) and unverified (unknown videos) fingerprints
- **Automatic Conflict Resolution**: Smart handling of potential duplicates
- **Batch Processing**: Analyze entire directories of videos

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
python smart_fingerprint_api.py
```

### 3. Analyze Videos
```bash
# Single video analysis
python smart_integration.py "path/to/video.mp4"

# Batch directory analysis
python smart_integration.py "path/to/videos/" --batch
```

### 4. Test the System
```bash
python smart_api_client.py
```

## 📁 Core Files

- **`smart_fingerprint_api.py`** - Main API server with smart search and verification
- **`smart_api_client.py`** - Client library with demonstration examples
- **`smart_integration.py`** - Integration script connecting local analysis with API
- **`video_fingerprinter.py`** - Core fingerprinting engine
- **`reset_database.py`** - Database management utility
- **`requirements.txt`** - Python dependencies

## 🔧 Technology Stack

- **OpenCV** - Video processing and computer vision
- **NumPy** - Numerical operations
- **Flask** - REST API server
- **SQLAlchemy** - Database management
- **SQLite** - Local database storage

## 📊 How It Works

### Fingerprinting Process
1. **Segment Analysis**: Divides video into 10-minute segments
2. **Grid Processing**: Each segment analyzed using 4x6 brightness grid (24 values)
3. **Encoding**: Converts brightness values to alphanumeric fingerprint codes
4. **Storage**: Saves fingerprints with metadata in database

### Smart Search Algorithm
1. **Name Matching**: Extract and compare clean filenames
2. **Exact Fingerprint**: Direct fingerprint code comparison
3. **Partial Matching**: Name similarity + segment comparison
4. **Confidence Scoring**: Different strategies have different confidence levels
5. **Verification Status**: Prioritizes verified over unverified matches

### Add Verified Fingerprint
1. **Conflict Detection**: Checks for existing similar fingerprints
2. **Name Analysis**: 50%+ similarity threshold with existing verified content
3. **Segment Comparison**: Visual similarity analysis for strong name matches
4. **Overwrite Options**: User choice for handling conflicts
5. **Automatic Cleanup**: Removes matching unverified duplicates

## 🌐 API Endpoints

- `POST /api/search` - Smart fingerprint search
- `POST /api/fingerprints` - Store new fingerprint
- `POST /api/fingerprints/add-verified` - Add verified fingerprint with conflict detection
- `PUT /api/fingerprints/<code>/verify` - Mark fingerprint as verified
- `GET /api/fingerprints/unverified` - List unverified fingerprints
- `GET /api/stats` - Database statistics

## 📚 Usage Examples

### Python API Usage
```python
from smart_api_client import SmartFingerprintAPIClient

client = SmartFingerprintAPIClient()

# Search for matches
result = client.search_fingerprint(
    fingerprint_code="ABCD1234EFGH5678IJKL9012MNOP3456-TEST",
    filename="video.mp4"
)

# Add verified fingerprint
result = client.add_verified_fingerprint(
    fingerprint_code="NEW123456789VIDEO987654321-VER",
    filename="new_video.mp4",
    title="My Video",
    media_type="movie"
)
```

### Command Line Usage
```bash
# Analyze and search for duplicates
python smart_integration.py "video.mp4"

# Batch process directory
python smart_integration.py "videos/" --batch --no-store
```

## 🔍 Duplicate Detection Types

- **Sequential Match**: Episodes found in correct order within compilation
- **Fragmented Match**: Episodes found out of order within compilation  
- **Exact Match**: Identical fingerprints (100% confidence)
- **Name Match**: High filename similarity with verified metadata
- **Visual Match**: Similar segments but different names (unknown videos)

This system provides a complete solution for video fingerprinting, duplicate detection, and content management with both local processing and online database capabilities.
