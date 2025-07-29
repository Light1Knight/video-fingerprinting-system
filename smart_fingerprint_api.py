#!/usr/bin/env python3
"""
Smart Video Fingerprint Search API
Takes a fingerprint, decodes it, and uses multiple search strategies to find matches.
"""
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
import os
import re
from difflib import SequenceMatcher

# Initialize Flask app
app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fingerprints.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-change-this'

db = SQLAlchemy(app)

# Database Models
class VideoFingerprint(db.Model):
    __tablename__ = 'video_fingerprints'
    
    id = db.Column(db.Integer, primary_key=True)
    fingerprint_code = db.Column(db.String(32), unique=True, nullable=False, index=True)
    filename = db.Column(db.String(255), nullable=False, index=True)
    extracted_name = db.Column(db.String(255), index=True)  # Cleaned name for matching
    duration = db.Column(db.Float, nullable=False)
    
    # Verification status
    is_verified = db.Column(db.Boolean, default=False, nullable=False, index=True)
    verification_source = db.Column(db.String(50))  # 'manual', 'auto', 'api', etc.
    
    # Metadata
    title = db.Column(db.String(255))
    season = db.Column(db.Integer)
    episode = db.Column(db.Integer)
    release_date = db.Column(db.String(20))
    media_type = db.Column(db.String(20))  # 'movie', 'tv_show', 'unknown'
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    segments = db.relationship('FingerprintSegment', backref='fingerprint', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self, include_segments=False):
        result = {
            'id': self.id,
            'fingerprint_code': self.fingerprint_code,
            'filename': self.filename,
            'extracted_name': self.extracted_name,
            'duration': self.duration,
            'is_verified': self.is_verified,
            'verification_source': self.verification_source,
            'title': self.title,
            'season': self.season,
            'episode': self.episode,
            'release_date': self.release_date,
            'media_type': self.media_type,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
        
        if include_segments:
            result['segments'] = [seg.to_dict() for seg in self.segments]
            
        return result

class FingerprintSegment(db.Model):
    __tablename__ = 'fingerprint_segments'
    
    id = db.Column(db.Integer, primary_key=True)
    fingerprint_id = db.Column(db.Integer, db.ForeignKey('video_fingerprints.id'), nullable=False)
    segment_index = db.Column(db.Integer, nullable=False)
    start_time = db.Column(db.Float, nullable=False)
    end_time = db.Column(db.Float, nullable=False)
    grid_averages = db.Column(db.Text, nullable=False)  # JSON string
    
    def to_dict(self):
        return {
            'segment_index': self.segment_index,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'grid_averages': json.loads(self.grid_averages)
        }

# Helper Functions
def extract_name_from_filename(filename):
    """Extract clean name from filename for matching."""
    # Remove file extension
    name = os.path.splitext(filename)[0]
    
    # Common patterns to clean
    patterns = [
        # Remove quality indicators
        r'\b(720p|1080p|4K|HD|BluRay|WEB-DL|HDTV|DVDRip)\b',
        # Remove codec info
        r'\b(x264|x265|H264|H265|HEVC|XviD)\b',
        # Remove audio info
        r'\b(AC3|DTS|AAC|MP3)\b',
        # Remove group names in brackets/parentheses at end
        r'[\[\(][^\]\)]*[\]\)]$',
        # Remove common separators and replace with spaces
        r'[._\-]+',
        # Remove extra whitespace
        r'\s+',
    ]
    
    for pattern in patterns[:-2]:  # Don't apply separator/whitespace patterns yet
        name = re.sub(pattern, '', name, flags=re.IGNORECASE)
    
    # Replace separators with spaces
    name = re.sub(r'[._\-]+', ' ', name)
    
    # Clean up whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

def decode_fingerprint(fingerprint_code):
    """Decode fingerprint to extract basic information."""
    if '-' in fingerprint_code:
        main_code, hash_suffix = fingerprint_code.split('-', 1)
    else:
        main_code = fingerprint_code
        hash_suffix = None
    
    # Basic decode - convert from base36-like to values
    decoded_values = []
    for char in main_code:
        if char.isdigit():
            decoded_values.append(int(char))
        else:
            decoded_values.append(ord(char.upper()) - ord('A') + 10)
    
    return {
        'main_code': main_code,
        'hash_suffix': hash_suffix,
        'decoded_values': decoded_values,
        'is_multi_segment': hash_suffix is not None
    }

def calculate_name_similarity(name1, name2):
    """Calculate similarity between two names."""
    if not name1 or not name2:
        return 0.0
    
    # Normalize for comparison
    norm1 = name1.lower().strip()
    norm2 = name2.lower().strip()
    
    # Exact match
    if norm1 == norm2:
        return 1.0
    
    # Use sequence matcher for partial similarity
    return SequenceMatcher(None, norm1, norm2).ratio()

def calculate_fingerprint_similarity(code1, code2):
    """Calculate similarity between two fingerprint codes."""
    if not code1 or not code2:
        return 0.0
    
    # Extract main codes
    main1 = code1.split('-')[0] if '-' in code1 else code1
    main2 = code2.split('-')[0] if '-' in code2 else code2
    
    if len(main1) != len(main2):
        return 0.0
    
    # Character-wise similarity
    matches = sum(1 for a, b in zip(main1, main2) if a == b)
    return matches / len(main1)

def compare_segments(segments1, segments2):
    """Compare segment data between two fingerprints."""
    if not segments1 or not segments2:
        return 0.0
    
    # Simple segment comparison - can be enhanced
    total_similarity = 0.0
    comparisons = 0
    
    for seg1 in segments1[:5]:  # Compare first 5 segments
        for seg2 in segments2[:5]:
            try:
                grid1 = json.loads(seg1.grid_averages) if isinstance(seg1.grid_averages, str) else seg1.grid_averages
                grid2 = json.loads(seg2.grid_averages) if isinstance(seg2.grid_averages, str) else seg2.grid_averages
                
                # Calculate grid similarity (simplified)
                flat1 = [item for sublist in grid1 for item in (sublist if isinstance(sublist, list) else [sublist])]
                flat2 = [item for sublist in grid2 for item in (sublist if isinstance(sublist, list) else [sublist])]
                
                if len(flat1) == len(flat2) and len(flat1) > 0:
                    # Calculate correlation-like similarity
                    diff = sum(abs(a - b) for a, b in zip(flat1, flat2))
                    max_diff = len(flat1) * 255  # Max possible difference
                    similarity = 1.0 - (diff / max_diff) if max_diff > 0 else 0.0
                    total_similarity += similarity
                    comparisons += 1
                    
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    
    return total_similarity / comparisons if comparisons > 0 else 0.0

# API Routes
@app.route('/')
def index():
    """API documentation."""
    return jsonify({
        'name': 'Smart Video Fingerprint Search API',
        'version': '1.0',
        'endpoints': {
            'POST /api/search': 'Smart search with fingerprint',
            'POST /api/fingerprints': 'Store a new fingerprint',
            'GET /api/stats': 'Get database statistics'
        }
    })

@app.route('/api/search', methods=['POST'])
def smart_search():
    """
    Smart search using multiple strategies:
    1. Extract name and look for exact name matches
    2. Look for exact fingerprint matches
    3. Look for partial name matches (20%-100%) and compare segments
    4. Look for partial fingerprint matches
    5. Return top 5 closest matches with confidence ratings
    """
    try:
        data = request.get_json()
        if not data or 'fingerprint_code' not in data:
            return jsonify({'error': 'fingerprint_code required'}), 400
        
        fingerprint_code = data['fingerprint_code']
        filename = data.get('filename', '')
        
        # Decode fingerprint
        decoded = decode_fingerprint(fingerprint_code)
        
        # Extract name from filename if provided
        extracted_name = extract_name_from_filename(filename) if filename else ''
        
        results = []
        search_steps = []
        
        # Step 1: Exact name matches (prioritize verified first)
        if extracted_name:
            search_steps.append("Step 1: Searching for exact name matches")
            # First check verified fingerprints
            verified_name_matches = VideoFingerprint.query.filter(
                VideoFingerprint.extracted_name.ilike(f'%{extracted_name}%'),
                VideoFingerprint.is_verified == True
            ).all()
            
            for match in verified_name_matches:
                similarity = calculate_name_similarity(extracted_name, match.extracted_name)
                if similarity >= 0.8:  # High name similarity
                    results.append({
                        'match': match.to_dict(),
                        'confidence': similarity * 0.95,  # Slight penalty for name-only match
                        'match_type': 'exact_name_verified',
                        'reason': f'Verified name similarity: {similarity:.3f}'
                    })
            
            # Then check unverified fingerprints
            unverified_name_matches = VideoFingerprint.query.filter(
                VideoFingerprint.extracted_name.ilike(f'%{extracted_name}%'),
                VideoFingerprint.is_verified == False
            ).all()
            
            for match in unverified_name_matches:
                similarity = calculate_name_similarity(extracted_name, match.extracted_name)
                if similarity >= 0.8:  # High name similarity
                    results.append({
                        'match': match.to_dict(),
                        'confidence': similarity * 0.75,  # Lower confidence for unverified
                        'match_type': 'exact_name_unverified',
                        'reason': f'Unverified name similarity: {similarity:.3f}'
                    })
        
        # Step 2: Exact fingerprint matches (prioritize verified)
        search_steps.append("Step 2: Searching for exact fingerprint matches")
        exact_fp_match = VideoFingerprint.query.filter_by(
            fingerprint_code=fingerprint_code
        ).first()
        
        if exact_fp_match:
            confidence = 1.0 if exact_fp_match.is_verified else 0.85
            match_type = 'exact_fingerprint_verified' if exact_fp_match.is_verified else 'exact_fingerprint_unverified'
            reason = 'Identical verified fingerprint' if exact_fp_match.is_verified else 'Identical unverified fingerprint (unknown video)'
            
            results.append({
                'match': exact_fp_match.to_dict(),
                'confidence': confidence,
                'match_type': match_type,
                'reason': reason
            })
        
        # Step 3: Partial name matches with segment comparison (check verified first)
        if extracted_name and len(results) < 5:
            search_steps.append("Step 3: Partial name matches with segment comparison")
            all_fingerprints = VideoFingerprint.query.order_by(VideoFingerprint.is_verified.desc()).all()
            
            for fp in all_fingerprints:
                if fp.fingerprint_code == fingerprint_code:
                    continue  # Skip exact matches already found
                
                name_sim = calculate_name_similarity(extracted_name, fp.extracted_name or '')
                if 0.2 <= name_sim < 0.8:  # Partial name match
                    # Get segments for comparison if available
                    segment_sim = 0.0
                    if hasattr(data, 'segments') and fp.segments:
                        # This would need segment data from the request
                        segment_sim = 0.5  # Placeholder - would compare actual segments
                    
                    combined_confidence = (name_sim * 0.6) + (segment_sim * 0.4)
                    
                    # Adjust confidence based on verification status
                    if not fp.is_verified:
                        combined_confidence *= 0.8  # Lower confidence for unverified
                    
                    if combined_confidence >= 0.3:
                        match_type = f'partial_name_with_segments_{"verified" if fp.is_verified else "unverified"}'
                        status_desc = "verified" if fp.is_verified else "unverified (unknown video)"
                        
                        results.append({
                            'match': fp.to_dict(),
                            'confidence': combined_confidence,
                            'match_type': match_type,
                            'reason': f'{status_desc} - Name: {name_sim:.3f}, Segments: {segment_sim:.3f}'
                        })
        
        # Step 4: Partial fingerprint matches (prioritize verified)
        if len(results) < 5:
            search_steps.append("Step 4: Searching for partial fingerprint matches")
            all_fingerprints = VideoFingerprint.query.order_by(VideoFingerprint.is_verified.desc()).all()
            
            for fp in all_fingerprints:
                if fp.fingerprint_code == fingerprint_code:
                    continue
                
                fp_sim = calculate_fingerprint_similarity(fingerprint_code, fp.fingerprint_code)
                if fp_sim >= 0.3:  # Minimum fingerprint similarity
                    base_confidence = fp_sim * 0.8  # Penalty for partial match
                    
                    # Further adjust based on verification status
                    if not fp.is_verified:
                        base_confidence *= 0.75  # Lower confidence for unverified
                    
                    match_type = f'partial_fingerprint_{"verified" if fp.is_verified else "unverified"}'
                    status_desc = "verified" if fp.is_verified else "unverified (unknown video)"
                    
                    results.append({
                        'match': fp.to_dict(),
                        'confidence': base_confidence,
                        'match_type': match_type,
                        'reason': f'{status_desc} - Fingerprint similarity: {fp_sim:.3f}'
                    })
        
        # Step 5: If no matches found, save as unverified
        if len(results) == 0:
            search_steps.append("Step 5: No matches found - saving as unverified fingerprint")
            
            # Create unverified fingerprint entry
            unverified_fp = VideoFingerprint(
                fingerprint_code=fingerprint_code,
                filename=filename or 'unknown.mp4',
                extracted_name=extracted_name or 'Unknown Video',
                duration=data.get('duration', 0),
                is_verified=False,
                verification_source='auto_unverified',
                media_type='unknown'
            )
            
            try:
                db.session.add(unverified_fp)
                
                # Add segments if provided
                if 'segments' in data:
                    for i, segment_data in enumerate(data['segments']):
                        segment = FingerprintSegment(
                            fingerprint_id=unverified_fp.id,
                            segment_index=i,
                            start_time=segment_data.get('start_time', 0),
                            end_time=segment_data.get('end_time', 0),
                            grid_averages=json.dumps(segment_data.get('grid_averages', []))
                        )
                        db.session.add(segment)
                
                db.session.commit()
                
                return jsonify({
                    'query': {
                        'fingerprint_code': fingerprint_code,
                        'filename': filename,
                        'extracted_name': extracted_name,
                        'decoded_info': decoded
                    },
                    'search_steps': search_steps,
                    'matches_found': 0,
                    'results': [],
                    'action_taken': 'saved_as_unverified',
                    'message': 'No matches found. Fingerprint saved as unverified for future matching.',
                    'unverified_fingerprint': unverified_fp.to_dict()
                })
                
            except Exception as e:
                db.session.rollback()
                search_steps.append(f"Failed to save unverified fingerprint: {str(e)}")
        
        # Sort by confidence and return top 5
        results.sort(key=lambda x: x['confidence'], reverse=True)
        top_results = results[:5]
        
        return jsonify({
            'query': {
                'fingerprint_code': fingerprint_code,
                'filename': filename,
                'extracted_name': extracted_name,
                'decoded_info': decoded
            },
            'search_steps': search_steps,
            'matches_found': len(top_results),
            'results': top_results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/fingerprints', methods=['POST'])
def store_fingerprint():
    """Store a new fingerprint in the database."""
    try:
        data = request.get_json()
        
        if not data or 'fingerprint_code' not in data:
            return jsonify({'error': 'fingerprint_code required'}), 400
        
        # Extract name from filename
        filename = data.get('filename', '')
        extracted_name = extract_name_from_filename(filename)
        
        # Check if already exists
        existing = VideoFingerprint.query.filter_by(
            fingerprint_code=data['fingerprint_code']
        ).first()
        
        if existing:
            return jsonify({
                'message': 'Fingerprint already exists',
                'fingerprint': existing.to_dict()
            }), 200
        
        # Determine verification status
        is_verified = data.get('is_verified', False)
        verification_source = data.get('verification_source', 'manual' if is_verified else 'auto')
        
        # Create new fingerprint
        fingerprint = VideoFingerprint(
            fingerprint_code=data['fingerprint_code'],
            filename=filename,
            extracted_name=extracted_name,
            duration=data.get('duration', 0),
            is_verified=is_verified,
            verification_source=verification_source,
            title=data.get('title'),
            season=data.get('season'),
            episode=data.get('episode'),
            release_date=data.get('release_date'),
            media_type=data.get('media_type', 'unknown')
        )
        
        db.session.add(fingerprint)
        db.session.flush()
        
        # Add segments if provided
        if 'segments' in data:
            for i, segment_data in enumerate(data['segments']):
                segment = FingerprintSegment(
                    fingerprint_id=fingerprint.id,
                    segment_index=i,
                    start_time=segment_data.get('start_time', 0),
                    end_time=segment_data.get('end_time', 0),
                    grid_averages=json.dumps(segment_data.get('grid_averages', []))
                )
                db.session.add(segment)
        
        db.session.commit()
        
        return jsonify({
            'message': 'Fingerprint stored successfully',
            'fingerprint': fingerprint.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics."""
    try:
        total = VideoFingerprint.query.count()
        verified = VideoFingerprint.query.filter_by(is_verified=True).count()
        unverified = VideoFingerprint.query.filter_by(is_verified=False).count()
        movies = VideoFingerprint.query.filter_by(media_type='movie').count()
        tv_shows = VideoFingerprint.query.filter_by(media_type='tv_show').count()
        unknown = VideoFingerprint.query.filter_by(media_type='unknown').count()
        
        return jsonify({
            'total_fingerprints': total,
            'verified_fingerprints': verified,
            'unverified_fingerprints': unverified,
            'movies': movies,
            'tv_shows': tv_shows,
            'unknown_type': unknown,
            'total_duration_hours': round(
                (db.session.query(db.func.sum(VideoFingerprint.duration)).scalar() or 0) / 3600, 2
            )
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/fingerprints/<fingerprint_code>/verify', methods=['PUT'])
def verify_fingerprint(fingerprint_code):
    """Mark a fingerprint as verified and add metadata."""
    try:
        data = request.get_json() or {}
        
        fingerprint = VideoFingerprint.query.filter_by(
            fingerprint_code=fingerprint_code
        ).first()
        
        if not fingerprint:
            return jsonify({'error': 'Fingerprint not found'}), 404
        
        # Update verification status and metadata
        fingerprint.is_verified = True
        fingerprint.verification_source = data.get('verification_source', 'manual')
        
        # Update metadata if provided
        if 'title' in data:
            fingerprint.title = data['title']
        if 'season' in data:
            fingerprint.season = data['season']
        if 'episode' in data:
            fingerprint.episode = data['episode']
        if 'release_date' in data:
            fingerprint.release_date = data['release_date']
        if 'media_type' in data:
            fingerprint.media_type = data['media_type']
        
        db.session.commit()
        
        return jsonify({
            'message': 'Fingerprint verified successfully',
            'fingerprint': fingerprint.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/fingerprints/unverified', methods=['GET'])
def get_unverified_fingerprints():
    """Get all unverified fingerprints for manual verification."""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        unverified = VideoFingerprint.query.filter_by(
            is_verified=False
        ).order_by(VideoFingerprint.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'unverified_fingerprints': [fp.to_dict() for fp in unverified.items],
            'total': unverified.total,
            'pages': unverified.pages,
            'current_page': page,
            'per_page': per_page
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/fingerprints/add-verified', methods=['POST'])
def add_verified_fingerprint():
    """
    Smart add verified fingerprint with conflict detection and cleanup.
    
    Process:
    1. Look for exact fingerprint match
    2. If not found, decode and check name similarity (50%+ match)
    3. If name match found, check segments for verification
    4. Handle overwrite decisions for name conflicts
    5. Clean up matching unverified fingerprints after adding
    """
    try:
        data = request.get_json()
        if not data or 'fingerprint_code' not in data:
            return jsonify({'error': 'fingerprint_code required'}), 400
        
        fingerprint_code = data['fingerprint_code']
        filename = data.get('filename', '')
        extracted_name = extract_name_from_filename(filename) if filename else ''
        
        # Decode fingerprint for analysis
        decoded = decode_fingerprint(fingerprint_code)
        
        response_data = {
            'fingerprint_code': fingerprint_code,
            'filename': filename,
            'extracted_name': extracted_name,
            'decoded_info': decoded,
            'steps_performed': [],
            'conflicts_found': [],
            'action_required': None,
            'result': None
        }
        
        # Step 1: Check for exact fingerprint match
        response_data['steps_performed'].append("Step 1: Checking for exact fingerprint match")
        
        existing_exact = VideoFingerprint.query.filter_by(
            fingerprint_code=fingerprint_code
        ).first()
        
        if existing_exact:
            if existing_exact.is_verified:
                return jsonify({
                    **response_data,
                    'result': 'already_exists_verified',
                    'message': 'Fingerprint already exists and is verified',
                    'existing_fingerprint': existing_exact.to_dict()
                }), 200
            else:
                # Unverified exists - we can upgrade it
                response_data['steps_performed'].append("Found existing unverified fingerprint - upgrading to verified")
                
                # Update the existing unverified fingerprint
                existing_exact.is_verified = True
                existing_exact.verification_source = data.get('verification_source', 'manual')
                
                # Update metadata if provided
                if 'title' in data:
                    existing_exact.title = data['title']
                if 'season' in data:
                    existing_exact.season = data['season']
                if 'episode' in data:
                    existing_exact.episode = data['episode']
                if 'release_date' in data:
                    existing_exact.release_date = data['release_date']
                if 'media_type' in data:
                    existing_exact.media_type = data['media_type']
                
                db.session.commit()
                
                return jsonify({
                    **response_data,
                    'result': 'upgraded_from_unverified',
                    'message': 'Existing unverified fingerprint upgraded to verified',
                    'verified_fingerprint': existing_exact.to_dict()
                }), 200
        
        # Step 2: Check for name similarity matches (50%+ threshold)
        response_data['steps_performed'].append("Step 2: Checking for name similarity matches (50%+ threshold)")
        
        if extracted_name:
            # Check all verified fingerprints for name similarity
            all_verified = VideoFingerprint.query.filter_by(is_verified=True).all()
            name_matches = []
            
            for fp in all_verified:
                if fp.extracted_name:
                    similarity = calculate_name_similarity(extracted_name, fp.extracted_name)
                    if similarity >= 0.5:  # 50% threshold
                        name_matches.append({
                            'fingerprint': fp,
                            'similarity': similarity,
                            'segment_match': None  # Will be calculated if needed
                        })
            
            if name_matches:
                response_data['steps_performed'].append(f"Found {len(name_matches)} name matches above 50% threshold")
                
                # Step 3: Check segments for the best name matches
                for match in name_matches:
                    if 'segments' in data and match['fingerprint'].segments:
                        # Compare segments (simplified comparison)
                        segment_similarity = compare_segments(
                            data.get('segments', []), 
                            match['fingerprint'].segments
                        )
                        match['segment_match'] = segment_similarity
                
                # Find the best match
                best_match = max(name_matches, key=lambda x: (x['similarity'], x.get('segment_match', 0)))
                
                # If we have a strong match (high name + segment similarity)
                if (best_match['similarity'] >= 0.8 and 
                    best_match.get('segment_match', 0) >= 0.7):
                    
                    response_data['conflicts_found'].append({
                        'type': 'strong_existing_match',
                        'existing_fingerprint': best_match['fingerprint'].to_dict(),
                        'name_similarity': best_match['similarity'],
                        'segment_similarity': best_match.get('segment_match', 0),
                        'confidence': (best_match['similarity'] + best_match.get('segment_match', 0)) / 2
                    })
                    
                    return jsonify({
                        **response_data,
                        'result': 'conflict_detected',
                        'action_required': 'confirm_overwrite',
                        'message': 'Strong match found with existing verified fingerprint. Confirm if you want to overwrite.',
                        'existing_match': best_match['fingerprint'].to_dict(),
                        'conflict_details': {
                            'name_similarity': round(best_match['similarity'], 3),
                            'segment_similarity': round(best_match.get('segment_match', 0), 3)
                        }
                    }), 409  # Conflict status
                
                # Moderate matches - list them for user review
                elif best_match['similarity'] >= 0.5:
                    response_data['conflicts_found'] = [{
                        'type': 'moderate_name_match',
                        'existing_fingerprint': match['fingerprint'].to_dict(),
                        'name_similarity': match['similarity'],
                        'segment_similarity': match.get('segment_match', 0)
                    } for match in name_matches if match['similarity'] >= 0.5]
                    
                    return jsonify({
                        **response_data,
                        'result': 'potential_conflicts',
                        'action_required': 'review_matches',
                        'message': f'Found {len(name_matches)} potential name matches. Review and confirm to proceed.',
                        'suggested_action': 'proceed_if_different' if best_match['similarity'] < 0.7 else 'confirm_overwrite'
                    }), 409
        
        # Step 4: No conflicts - proceed with adding verified fingerprint
        response_data['steps_performed'].append("Step 4: No conflicts found - adding verified fingerprint")
        
        # Handle overwrite parameter for forced operations
        force_overwrite = data.get('force_overwrite', False)
        overwrite_target = data.get('overwrite_fingerprint_id')
        
        if force_overwrite and overwrite_target:
            response_data['steps_performed'].append(f"Force overwrite requested for fingerprint ID: {overwrite_target}")
            
            # Delete the target fingerprint
            target_fp = VideoFingerprint.query.get(overwrite_target)
            if target_fp:
                db.session.delete(target_fp)
                response_data['steps_performed'].append(f"Deleted existing fingerprint: {target_fp.filename}")
        
        # Create the new verified fingerprint
        new_fingerprint = VideoFingerprint(
            fingerprint_code=fingerprint_code,
            filename=filename,
            extracted_name=extracted_name,
            duration=data.get('duration', 0),
            is_verified=True,
            verification_source=data.get('verification_source', 'manual'),
            title=data.get('title'),
            season=data.get('season'),
            episode=data.get('episode'),
            release_date=data.get('release_date'),
            media_type=data.get('media_type', 'unknown')
        )
        
        db.session.add(new_fingerprint)
        db.session.flush()  # Get the ID
        
        # Add segments if provided
        if 'segments' in data:
            for i, segment_data in enumerate(data['segments']):
                segment = FingerprintSegment(
                    fingerprint_id=new_fingerprint.id,
                    segment_index=i,
                    start_time=segment_data.get('start_time', 0),
                    end_time=segment_data.get('end_time', 0),
                    grid_averages=json.dumps(segment_data.get('grid_averages', []))
                )
                db.session.add(segment)
        
        # Step 5: Clean up matching unverified fingerprints
        response_data['steps_performed'].append("Step 5: Cleaning up matching unverified fingerprints")
        
        # Find unverified fingerprints that match this new verified one
        cleanup_candidates = []
        unverified_fps = VideoFingerprint.query.filter_by(is_verified=False).all()
        
        for unverified_fp in unverified_fps:
            should_delete = False
            delete_reason = ""
            
            # Check fingerprint similarity
            fp_similarity = calculate_fingerprint_similarity(
                fingerprint_code, unverified_fp.fingerprint_code
            )
            
            if fp_similarity >= 0.8:  # High fingerprint similarity
                should_delete = True
                delete_reason = f"High fingerprint similarity: {fp_similarity:.3f}"
            
            # Check name similarity if we have names
            elif extracted_name and unverified_fp.extracted_name:
                name_similarity = calculate_name_similarity(
                    extracted_name, unverified_fp.extracted_name
                )
                if name_similarity >= 0.8:  # High name similarity
                    should_delete = True
                    delete_reason = f"High name similarity: {name_similarity:.3f}"
            
            if should_delete:
                cleanup_candidates.append({
                    'fingerprint': unverified_fp.to_dict(),
                    'reason': delete_reason
                })
                db.session.delete(unverified_fp)
        
        # Commit all changes
        db.session.commit()
        
        response_data['steps_performed'].append(f"Successfully added verified fingerprint")
        if cleanup_candidates:
            response_data['steps_performed'].append(f"Cleaned up {len(cleanup_candidates)} matching unverified fingerprints")
        
        return jsonify({
            **response_data,
            'result': 'success',
            'message': 'Verified fingerprint added successfully',
            'verified_fingerprint': new_fingerprint.to_dict(),
            'cleanup_performed': cleanup_candidates,
            'cleanup_count': len(cleanup_candidates)
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/fingerprints/add-verified/confirm', methods=['POST'])
def confirm_verified_fingerprint():
    """
    Confirm adding a verified fingerprint after conflict resolution.
    Used when the initial add-verified call returned a conflict.
    """
    try:
        data = request.get_json()
        if not data or 'fingerprint_code' not in data:
            return jsonify({'error': 'fingerprint_code required'}), 400
        
        action = data.get('action')  # 'overwrite', 'proceed_anyway', 'cancel'
        
        if action == 'overwrite':
            # Add force_overwrite flag and retry
            data['force_overwrite'] = True
            return add_verified_fingerprint()
        
        elif action == 'proceed_anyway':
            # Proceed without overwriting, even with conflicts
            data['ignore_conflicts'] = True
            return add_verified_fingerprint()
        
        elif action == 'cancel':
            return jsonify({
                'result': 'cancelled',
                'message': 'Operation cancelled by user'
            }), 200
        
        else:
            return jsonify({'error': 'Invalid action. Use: overwrite, proceed_anyway, or cancel'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def create_tables():
    """Create database tables."""
    with app.app_context():
        db.create_all()
        print("Database tables created successfully!")

if __name__ == '__main__':
    create_tables()
    print("🗄️  Database initialized!")
    print("🚀 Starting Smart Video Fingerprint Search API...")
    print("📖 API documentation available at: http://localhost:5000/")
    app.run(host='0.0.0.0', port=5000, debug=True)
