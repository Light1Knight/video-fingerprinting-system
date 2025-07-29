#!/usr/bin/env python3
"""
Custom Video Fingerprinting System
Creates alphanumeric fingerprints based on 10-minute segments analyzed in 24 grid sections
"""
import cv2
import numpy as np
import os
import json
import re
from pathlib import Path
import hashlib
import string
import time
from datetime import datetime


class VideoFingerprinter:
    def __init__(self):
        """Initialize the video fingerprinter."""
        self.segment_duration = 600  # 10 minutes in seconds
        self.grid_rows = 4  # 4 rows of grid segments
        self.grid_cols = 6  # 6 columns of grid segments (4x6 = 24 segments)
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        
    def extract_fingerprint(self, video_path):
        """
        Extract a custom fingerprint from video using 10-minute segments.
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            dict: Complete fingerprint data
        """
        print(f"🎬 Creating custom fingerprint: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file")
            return None
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 Video info: {width}x{height}, {duration:.1f}s, {total_frames} frames")
        
        if duration < 60:  # Less than 1 minute
            print(f"⚠️  Video too short for reliable fingerprinting")
            cap.release()
            return None
        
        # Calculate how many 10-minute segments we can analyze
        num_segments = max(1, int(duration // self.segment_duration))
        if duration < self.segment_duration:
            # For videos shorter than 10 minutes, use the entire video as one segment
            segment_starts = [0]
            segment_durations = [duration]
        else:
            # For longer videos, use multiple 10-minute segments
            segment_starts = []
            segment_durations = []
            for i in range(num_segments):
                start_time = i * self.segment_duration
                if start_time + self.segment_duration <= duration:
                    segment_starts.append(start_time)
                    segment_durations.append(self.segment_duration)
                else:
                    # Last segment might be shorter
                    remaining = duration - start_time
                    if remaining >= 60:  # Only include if at least 1 minute
                        segment_starts.append(start_time)
                        segment_durations.append(remaining)
        
        print(f"🔍 Analyzing {len(segment_starts)} segments of video")
        
        # Extract fingerprint for each segment
        segment_fingerprints = []
        for i, (start_time, seg_duration) in enumerate(zip(segment_starts, segment_durations)):
            print(f"  📍 Processing segment {i+1}/{len(segment_starts)} ({start_time/60:.1f}-{(start_time+seg_duration)/60:.1f} min)")
            segment_fp = self._extract_segment_fingerprint(cap, start_time, seg_duration, fps, width, height)
            if segment_fp:
                segment_fingerprints.append(segment_fp)
        
        cap.release()
        
        if not segment_fingerprints:
            print(f"❌ Could not extract any segment fingerprints")
            return None
        
        # Generate overall fingerprint code
        fingerprint_code = self._generate_fingerprint_code(segment_fingerprints)
        
        # Extract initial title from filename (non-verified)
        initial_title = self._extract_title_from_filename(os.path.basename(video_path))
        
        # Create complete fingerprint
        fingerprint = {
            'file_path': video_path,
            'filename': os.path.basename(video_path),
            'duration': duration,
            'resolution': f"{width}x{height}",
            'fps': fps,
            'file_size': os.path.getsize(video_path),
            'segments_analyzed': len(segment_fingerprints),
            'segment_fingerprints': segment_fingerprints,
            'fingerprint_code': fingerprint_code,
            'created_at': datetime.now().isoformat(),
            'fingerprint_version': '1.0',
            # Title metadata with verification status
            'title_info': {
                'current_title': initial_title,
                'verified': False,
                'source': 'filename_extraction',
                'confidence': 0.3,  # Low confidence for filename-based titles
                'verification_history': [
                    {
                        'title': initial_title,
                        'source': 'filename_extraction',
                        'verified': False,
                        'confidence': 0.3,
                        'timestamp': datetime.now().isoformat()
                    }
                ]
            }
        }
        
        print(f"✅ Fingerprint created: {fingerprint_code}")
        return fingerprint
    
    def _extract_segment_fingerprint(self, cap, start_time, duration, fps, width, height):
        """
        Extract fingerprint for a single 10-minute segment.
        
        Args:
            cap: OpenCV VideoCapture object
            start_time (float): Start time in seconds
            duration (float): Duration of segment in seconds
            fps (float): Frames per second
            width (int): Video width
            height (int): Video height
            
        Returns:
            dict: Segment fingerprint data
        """
        # Calculate frame range for this segment
        start_frame = int(start_time * fps)
        end_frame = int((start_time + duration) * fps)
        total_segment_frames = end_frame - start_frame
        
        # Sample frames throughout the segment (1 frame per 5 seconds for efficiency)
        sample_interval = max(1, int(5 * fps))  # Sample every 5 seconds
        sample_frames = list(range(start_frame, end_frame, sample_interval))
        
        if not sample_frames:
            return None
        
        # Initialize grid accumulators for 24 segments (4 rows x 6 columns)
        grid_accumulators = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float64)
        frames_processed = 0
        
        # Calculate grid segment dimensions
        segment_height = height // self.grid_rows
        segment_width = width // self.grid_cols
        
        for frame_idx in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Convert to grayscale for black/white analysis
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Analyze each of the 24 grid segments
            for row in range(self.grid_rows):
                for col in range(self.grid_cols):
                    # Calculate segment boundaries
                    y_start = row * segment_height
                    y_end = min((row + 1) * segment_height, height)
                    x_start = col * segment_width
                    x_end = min((col + 1) * segment_width, width)
                    
                    # Extract segment and calculate average brightness
                    segment = gray_frame[y_start:y_end, x_start:x_end]
                    avg_brightness = np.mean(segment)
                    
                    # Accumulate brightness values
                    grid_accumulators[row, col] += avg_brightness
            
            frames_processed += 1
        
        if frames_processed == 0:
            return None
        
        # Calculate average brightness for each grid segment across all frames
        grid_averages = grid_accumulators / frames_processed
        
        # Generate segment code (24 characters for 24 grid segments)
        segment_code = self._grid_to_code(grid_averages)
        
        return {
            'start_time': start_time,
            'duration': duration,
            'frames_analyzed': frames_processed,
            'grid_averages': grid_averages.tolist(),
            'segment_code': segment_code
        }
    
    def _grid_to_code(self, grid_averages):
        """
        Convert 4x6 grid of brightness averages to 24-character alphanumeric code.
        
        Args:
            grid_averages (np.array): 4x6 array of brightness averages (0-255)
            
        Returns:
            str: 24-character alphanumeric code
        """
        # Define character set (36 characters: 0-9, A-Z)
        charset = string.digits + string.ascii_uppercase  # 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
        
        code_chars = []
        
        # Convert each grid segment to a character
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                brightness = grid_averages[row, col]
                
                # Map brightness (0-255) to character index (0-35)
                # Divide brightness into 36 levels
                char_index = min(35, int(brightness * 36 / 256))
                code_chars.append(charset[char_index])
        
        return ''.join(code_chars)
    
    def _generate_fingerprint_code(self, segment_fingerprints):
        """
        Generate overall fingerprint code from all segments.
        
        Args:
            segment_fingerprints (list): List of segment fingerprint data
            
        Returns:
            str: Complete fingerprint code
        """
        if not segment_fingerprints:
            return ""
        
        if len(segment_fingerprints) == 1:
            # Single segment - use its code directly
            return segment_fingerprints[0]['segment_code']
        
        # Multiple segments - combine them with separators
        segment_codes = [seg['segment_code'] for seg in segment_fingerprints]
        
        # For multiple segments, create a composite code
        # Use first segment + hash of all segments for uniqueness
        first_segment = segment_codes[0]
        
        # Create hash of all segment codes combined
        all_codes_combined = ''.join(segment_codes)
        hash_object = hashlib.md5(all_codes_combined.encode())
        hash_hex = hash_object.hexdigest()
        
        # Convert first 8 chars of hash to our character set
        charset = string.digits + string.ascii_uppercase
        hash_code = ""
        for i in range(0, 8, 2):
            hex_pair = hash_hex[i:i+2]
            decimal_val = int(hex_pair, 16)
            char_index = decimal_val % 36
            hash_code += charset[char_index]
        
        # Combine: first segment (24 chars) + separator + hash (4 chars)
        return f"{first_segment}-{hash_code}"
    
    def _extract_title_from_filename(self, filename):
        """
        Extract a potential title from the filename.
        
        Args:
            filename (str): Original filename
            
        Returns:
            str: Extracted title
        """
        # Remove file extension
        name_without_ext = Path(filename).stem
        
        # Common patterns to extract show/movie names
        title_patterns = [
            # TV Show patterns
            r'^(.+?)[\s\._\-]S\d+E\d+',  # ShowName S01E01
            r'^(.+?)[\s\._\-]Season\s+\d+',  # ShowName Season 1
            r'^(.+?)[\s\._\-]\d+x\d+',  # ShowName 1x01
            # DVD patterns
            r'^(.+?)\s+SEASON\s+\d+\s+DISC',  # SHOWNAME SEASON 1 DISC
            r'^(.+?)\s+DISC\s+\d+',  # SHOWNAME DISC 1
            # Movie patterns with year
            r'^(.+?)[\s\._\-]\(\d{4}\)',  # MovieName (2021)
            r'^(.+?)[\s\._\-]\d{4}[\s\._\-]',  # MovieName 2021
            # General cleanup - remove common suffixes
            r'^(.+?)[\s\._\-](BluRay|BDRip|DVDRip|WEBRip|720p|1080p|x264|x265)',
        ]
        
        extracted_title = None
        
        # Try each pattern
        for pattern in title_patterns:
            match = re.search(pattern, name_without_ext, re.IGNORECASE)
            if match:
                extracted_title = match.group(1).strip()
                break
        
        # If no pattern matched, use the whole filename (cleaned up)
        if not extracted_title:
            extracted_title = name_without_ext
        
        # Clean up the title
        extracted_title = self._clean_title(extracted_title)
        
        return extracted_title if extracted_title else "Unknown Title"
    
    def _clean_title(self, title):
        """
        Clean up extracted title for better readability.
        
        Args:
            title (str): Raw extracted title
            
        Returns:
            str: Cleaned title
        """
        if not title:
            return "Unknown Title"
        
        # Remove technical suffixes and disc information
        title = re.sub(r'[_\-]t\d+$', '', title)  # Remove _t00, -t01 etc
        title = re.sub(r'[_\-][A-Z]\d+$', '', title)  # Remove -B1, _C2 etc
        
        # Replace dots, underscores, multiple dashes with spaces
        title = re.sub(r'[\._]+', ' ', title)
        title = re.sub(r'[-]{2,}', ' ', title)
        
        # Clean up spacing
        title = ' '.join(title.split())
        
        # Convert to title case
        title = title.title()
        
        # Handle common abbreviations
        title = re.sub(r'\bS(\d+)\b', r'Season \1', title)
        title = re.sub(r'\bE(\d+)\b', r'Episode \1', title)
        
        return title.strip()
    
    def update_title_verification(self, fingerprint, new_title, source, verified=True, confidence=0.9):
        """
        Update the title information in a fingerprint with verification data.
        
        Args:
            fingerprint (dict): Fingerprint to update
            new_title (str): New verified title
            source (str): Source of verification (e.g., 'TMDB', 'OMDB', 'manual')
            verified (bool): Whether this title is verified
            confidence (float): Confidence level (0.0-1.0)
            
        Returns:
            dict: Updated fingerprint
        """
        if not fingerprint or 'title_info' not in fingerprint:
            return fingerprint
        
        # Add to verification history
        verification_entry = {
            'title': new_title,
            'source': source,
            'verified': verified,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        fingerprint['title_info']['verification_history'].append(verification_entry)
        
        # Update current title if this is more reliable
        current_confidence = fingerprint['title_info'].get('confidence', 0.0)
        if confidence > current_confidence or (verified and not fingerprint['title_info']['verified']):
            fingerprint['title_info']['current_title'] = new_title
            fingerprint['title_info']['verified'] = verified
            fingerprint['title_info']['source'] = source
            fingerprint['title_info']['confidence'] = confidence
        
        return fingerprint
    
    def get_title_info(self, fingerprint):
        """
        Get title information from fingerprint.
        
        Args:
            fingerprint (dict): Fingerprint data
            
        Returns:
            dict: Title information
        """
        if not fingerprint or 'title_info' not in fingerprint:
            return {
                'title': 'Unknown',
                'verified': False,
                'source': 'unknown',
                'confidence': 0.0
            }
        
        title_info = fingerprint['title_info']
        return {
            'title': title_info.get('current_title', 'Unknown'),
            'verified': title_info.get('verified', False),
            'source': title_info.get('source', 'unknown'),
            'confidence': title_info.get('confidence', 0.0),
            'history_count': len(title_info.get('verification_history', []))
        }
    
    def compare_fingerprints(self, fp1, fp2, threshold=0.85):
        """
        Compare two fingerprints for similarity with advanced duplicate detection.
        
        Args:
            fp1 (dict): First fingerprint
            fp2 (dict): Second fingerprint
            threshold (float): Similarity threshold (0.0-1.0)
            
        Returns:
            dict: Comparison result with detailed analysis
        """
        if not fp1 or not fp2:
            return {'similar': False, 'similarity': 0.0, 'reason': 'Invalid fingerprints'}
        
        # Quick check - same fingerprint code means identical
        if fp1['fingerprint_code'] == fp2['fingerprint_code']:
            return {
                'similar': True, 
                'similarity': 1.0, 
                'reason': 'Identical fingerprint codes',
                'match_type': 'identical'
            }
        
        # Get basic info
        duration1 = fp1.get('duration', 0)
        duration2 = fp2.get('duration', 0)
        segments1 = fp1.get('segment_fingerprints', [])
        segments2 = fp2.get('segment_fingerprints', [])
        
        if not segments1 or not segments2:
            return {'similar': False, 'similarity': 0.0, 'reason': 'No segment data'}
        
        # Determine which video is longer
        if duration1 > duration2 * 1.5:  # fp1 is significantly longer
            longer_fp, shorter_fp = fp1, fp2
            longer_segments, shorter_segments = segments1, segments2
            longer_is_first = True
        elif duration2 > duration1 * 1.5:  # fp2 is significantly longer
            longer_fp, shorter_fp = fp2, fp1
            longer_segments, shorter_segments = segments2, segments1
            longer_is_first = False
        else:
            # Similar lengths - use standard comparison
            return self._compare_similar_length_videos(fp1, fp2, threshold)
        
        # Advanced comparison for different length videos
        return self._compare_different_length_videos(
            longer_fp, shorter_fp, longer_segments, shorter_segments, 
            threshold, longer_is_first
        )
    
    def _compare_similar_length_videos(self, fp1, fp2, threshold):
        """Compare videos of similar length."""
        duration1 = fp1.get('duration', 0)
        duration2 = fp2.get('duration', 0)
        segments1 = fp1.get('segment_fingerprints', [])
        segments2 = fp2.get('segment_fingerprints', [])
        
        # Duration similarity check
        duration_diff = abs(duration1 - duration2)
        max_duration = max(duration1, duration2)
        duration_similarity = 1.0 - (duration_diff / max_duration) if max_duration > 0 else 0.0
        
        # If durations are very different, likely different videos
        if duration_similarity < 0.9:
            return {
                'similar': False, 
                'similarity': duration_similarity, 
                'reason': 'Duration mismatch',
                'match_type': 'different_content'
            }
        
        # Compare segment fingerprints (standard method)
        seg1 = segments1[0] if segments1 else None
        seg2 = segments2[0] if segments2 else None
        
        if not seg1 or not seg2:
            return {'similar': False, 'similarity': 0.0, 'reason': 'Missing segment data'}
        
        grid1 = np.array(seg1['grid_averages'])
        grid2 = np.array(seg2['grid_averages'])
        
        # Calculate grid similarity using correlation coefficient
        correlation = np.corrcoef(grid1.flatten(), grid2.flatten())[0, 1]
        
        # Handle NaN correlation
        if np.isnan(correlation):
            correlation = 0.0
        
        # Convert correlation to similarity (0-1 range)
        similarity = (correlation + 1) / 2
        
        # Weight the final similarity
        final_similarity = (similarity * 0.8) + (duration_similarity * 0.2)
        
        return {
            'similar': final_similarity >= threshold,
            'similarity': final_similarity,
            'grid_correlation': correlation,
            'duration_similarity': duration_similarity,
            'match_type': 'standard_comparison',
            'reason': f'Grid correlation: {correlation:.3f}, Duration match: {duration_similarity:.3f}'
        }
    
    def _compare_different_length_videos(self, longer_fp, shorter_fp, longer_segments, shorter_segments, threshold, longer_is_first):
        """
        Compare videos of different lengths to detect if shorter video is contained in longer one.
        """
        # Get the number of segments to compare
        num_shorter_segments = len(shorter_segments)
        num_longer_segments = len(longer_segments)
        
        if num_shorter_segments == 0 or num_longer_segments == 0:
            return {'similar': False, 'similarity': 0.0, 'reason': 'No segments to compare'}
        
        # Convert segments to comparison grids
        shorter_grids = [np.array(seg['grid_averages']) for seg in shorter_segments]
        longer_grids = [np.array(seg['grid_averages']) for seg in longer_segments]
        
        # Find the best matching sequence
        best_match = self._find_best_sequence_match(longer_grids, shorter_grids)
        
        # Analyze the match
        if best_match['sequential_similarity'] >= threshold:
            match_type = 'sequential_duplicate'
            similarity = best_match['sequential_similarity']
            reason = f"Sequential match found: segments {best_match['start_index']}-{best_match['end_index']} in longer video"
            
        elif best_match['fragmented_similarity'] >= threshold * 0.9:  # Slightly lower threshold for fragmented
            match_type = 'fragmented_duplicate'
            similarity = best_match['fragmented_similarity']
            reason = f"Fragmented match found: {best_match['matched_segments']} segments matched"
            
        else:
            match_type = 'different_content'
            similarity = max(best_match['sequential_similarity'], best_match['fragmented_similarity'])
            reason = f"No significant match found (best: {similarity:.3f})"
        
        # Duration ratio for context
        duration_ratio = longer_fp['duration'] / shorter_fp['duration'] if shorter_fp['duration'] > 0 else 0
        
        return {
            'similar': similarity >= threshold,
            'similarity': similarity,
            'match_type': match_type,
            'reason': reason,
            'duration_ratio': duration_ratio,
            'longer_is_first': longer_is_first,
            'match_details': {
                'sequential_similarity': best_match['sequential_similarity'],
                'fragmented_similarity': best_match['fragmented_similarity'],
                'best_sequence_start': best_match['start_index'],
                'best_sequence_end': best_match['end_index'],
                'matched_segment_count': best_match['matched_segments']
            }
        }
    
    def _find_best_sequence_match(self, longer_grids, shorter_grids):
        """
        Find the best matching sequence of segments between longer and shorter videos.
        
        Args:
            longer_grids (list): Grid arrays from longer video
            shorter_grids (list): Grid arrays from shorter video
            
        Returns:
            dict: Best match information
        """
        num_longer = len(longer_grids)
        num_shorter = len(shorter_grids)
        
        best_sequential_similarity = 0.0
        best_start_index = 0
        best_end_index = 0
        
        # Try to find sequential matches (sliding window)
        for start_idx in range(num_longer - num_shorter + 1):
            end_idx = start_idx + num_shorter
            
            # Compare the sequence
            similarities = []
            for i in range(num_shorter):
                longer_grid = longer_grids[start_idx + i]
                shorter_grid = shorter_grids[i]
                
                # Calculate correlation between grid segments
                correlation = np.corrcoef(longer_grid.flatten(), shorter_grid.flatten())[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
                
                # Convert to similarity (0-1 range)
                similarity = (correlation + 1) / 2
                similarities.append(similarity)
            
            # Average similarity for this sequence
            sequence_similarity = np.mean(similarities)
            
            if sequence_similarity > best_sequential_similarity:
                best_sequential_similarity = sequence_similarity
                best_start_index = start_idx
                best_end_index = end_idx - 1
        
        # Check for fragmented matches (best individual segment matches)
        fragmented_similarities = []
        for shorter_grid in shorter_grids:
            best_segment_similarity = 0.0
            
            for longer_grid in longer_grids:
                correlation = np.corrcoef(shorter_grid.flatten(), longer_grid.flatten())[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
                
                similarity = (correlation + 1) / 2
                if similarity > best_segment_similarity:
                    best_segment_similarity = similarity
            
            fragmented_similarities.append(best_segment_similarity)
        
        # Count how many segments have good matches (>0.7)
        good_matches = sum(1 for sim in fragmented_similarities if sim > 0.7)
        fragmented_similarity = np.mean(fragmented_similarities)
        
        return {
            'sequential_similarity': best_sequential_similarity,
            'fragmented_similarity': fragmented_similarity,
            'start_index': best_start_index,
            'end_index': best_end_index,
            'matched_segments': good_matches
        }
    
    def detect_duplicates(self, video_paths, threshold=0.85):
        """
        Detect duplicates among a collection of videos.
        
        Args:
            video_paths (list): List of video file paths to analyze
            threshold (float): Similarity threshold for duplicate detection
            
        Returns:
            dict: Duplicate analysis results
        """
        print("🔍 Starting duplicate detection analysis...")
        
        # Generate fingerprints for all videos
        fingerprints = {}
        for video_path in video_paths:
            print(f"  Processing: {os.path.basename(video_path)}")
            try:
                fp = self.extract_fingerprint(video_path)
                if fp:
                    fingerprints[video_path] = fp
            except Exception as e:
                print(f"  ⚠️ Error processing {video_path}: {e}")
        
        if len(fingerprints) < 2:
            return {'duplicates': [], 'analysis': 'Need at least 2 valid videos to compare'}
        
        # Compare all pairs
        duplicates = []
        comparisons = []
        video_list = list(fingerprints.keys())
        
        for i in range(len(video_list)):
            for j in range(i + 1, len(video_list)):
                video1 = video_list[i]
                video2 = video_list[j]
                
                comparison = self.compare_fingerprints(
                    fingerprints[video1], 
                    fingerprints[video2], 
                    threshold
                )
                
                comparison_result = {
                    'video1': video1,
                    'video2': video2,
                    'video1_duration': fingerprints[video1]['duration'],
                    'video2_duration': fingerprints[video2]['duration'],
                    **comparison
                }
                
                comparisons.append(comparison_result)
                
                if comparison['similar']:
                    duplicates.append(comparison_result)
        
        # Analyze duplicate groups
        duplicate_groups = self._group_duplicates(duplicates)
        
        # Generate summary
        summary = {
            'total_videos': len(fingerprints),
            'total_comparisons': len(comparisons),
            'duplicate_pairs': len(duplicates),
            'duplicate_groups': len(duplicate_groups),
            'duplicates': duplicates,
            'groups': duplicate_groups,
            'all_comparisons': comparisons
        }
        
        return summary
    
    def _group_duplicates(self, duplicates):
        """
        Group duplicate pairs into connected components.
        
        Args:
            duplicates (list): List of duplicate pair results
            
        Returns:
            list: List of duplicate groups
        """
        if not duplicates:
            return []
        
        # Create a graph of connections
        connections = {}
        for dup in duplicates:
            video1 = dup['video1']
            video2 = dup['video2']
            
            if video1 not in connections:
                connections[video1] = set()
            if video2 not in connections:
                connections[video2] = set()
            
            connections[video1].add(video2)
            connections[video2].add(video1)
        
        # Find connected components (groups)
        visited = set()
        groups = []
        
        for video in connections:
            if video not in visited:
                group = self._find_connected_videos(video, connections, visited)
                if len(group) > 1:
                    # Sort by duration (longest first) and add metadata
                    group_info = []
                    for v in group:
                        # Find this video's info from duplicates
                        duration = None
                        for dup in duplicates:
                            if dup['video1'] == v:
                                duration = dup['video1_duration']
                                break
                            elif dup['video2'] == v:
                                duration = dup['video2_duration']
                                break
                        
                        group_info.append({
                            'video': v,
                            'filename': os.path.basename(v),
                            'duration': duration or 0
                        })
                    
                    # Sort by duration (longest first)
                    group_info.sort(key=lambda x: x['duration'], reverse=True)
                    groups.append(group_info)
        
        return groups
    
    def _find_connected_videos(self, start_video, connections, visited):
        """Find all videos connected to the start video."""
        group = [start_video]
        visited.add(start_video)
        
        for connected_video in connections[start_video]:
            if connected_video not in visited:
                group.extend(self._find_connected_videos(connected_video, connections, visited))
        
        return group
    
    def print_duplicate_report(self, duplicate_analysis):
        """
        Print a detailed duplicate detection report.
        
        Args:
            duplicate_analysis (dict): Results from detect_duplicates()
        """
        print("\n" + "="*60)
        print("📊 DUPLICATE DETECTION REPORT")
        print("="*60)
        
        # Handle case where analysis failed
        if 'analysis' in duplicate_analysis and 'total_videos' not in duplicate_analysis:
            print(f"Analysis result: {duplicate_analysis['analysis']}")
            return
        
        print(f"Total videos analyzed: {duplicate_analysis.get('total_videos', 0)}")
        print(f"Total comparisons made: {duplicate_analysis.get('total_comparisons', 0)}")
        print(f"Duplicate pairs found: {duplicate_analysis.get('duplicate_pairs', 0)}")
        print(f"Duplicate groups found: {duplicate_analysis.get('duplicate_groups', 0)}")
        
        if duplicate_analysis.get('duplicate_groups', 0) == 0:
            print("\n✅ No duplicates detected!")
            return
        
        print(f"\n🔍 DUPLICATE GROUPS:")
        print("-" * 40)
        
        for i, group in enumerate(duplicate_analysis.get('groups', []), 1):
            print(f"\nGroup {i} ({len(group)} videos):")
            
            for j, video_info in enumerate(group):
                duration_str = f"{video_info['duration']//60:.0f}m {video_info['duration']%60:.0f}s"
                marker = "🎬" if j == 0 else "📄"  # Mark longest as primary
                print(f"  {marker} {video_info['filename']} ({duration_str})")
        
        print(f"\n🔍 DETAILED ANALYSIS:")
        print("-" * 40)
        
        for dup in duplicate_analysis.get('duplicates', []):
            file1 = os.path.basename(dup['video1'])
            file2 = os.path.basename(dup['video2'])
            similarity = dup['similarity']
            match_type = dup.get('match_type', 'standard_comparison')
            
            duration1 = dup['video1_duration']
            duration2 = dup['video2_duration']
            
            print(f"\n📋 {file1} ↔ {file2}")
            print(f"    Similarity: {similarity:.3f} ({match_type})")
            print(f"    Durations: {duration1//60:.0f}m{duration1%60:.0f}s ↔ {duration2//60:.0f}m{duration2%60:.0f}s")
            print(f"    Reason: {dup['reason']}")
            
            if 'match_details' in dup:
                details = dup['match_details']
                print(f"    Sequential match: {details['sequential_similarity']:.3f}")
                print(f"    Fragmented match: {details['fragmented_similarity']:.3f}")
                if details['sequential_similarity'] > 0.5:
                    print(f"    Best sequence: segments {details['best_sequence_start']}-{details['best_sequence_end']}")
        
        print("\n" + "="*60)

    def organize_duplicates(self, duplicate_analysis, base_directory=None, move_files=False):
        """
        Organize duplicate files by moving compilation videos to a separate folder.
        
        Args:
            duplicate_analysis (dict): Results from detect_duplicates()
            base_directory (str): Base directory for organization (defaults to first video's directory)
            move_files (bool): Whether to actually move files or just plan the moves
            
        Returns:
            dict: Organization plan and results
        """
        if not duplicate_analysis.get('groups'):
            return {'message': 'No duplicate groups found to organize'}
        
        organization_plan = {
            'compilation_moves': [],
            'individual_keeps': [],
            'errors': []
        }
        
        # Determine base directory
        if not base_directory and duplicate_analysis.get('duplicates'):
            first_video = duplicate_analysis['duplicates'][0]['video1']
            base_directory = os.path.dirname(first_video)
        
        if not base_directory:
            organization_plan['errors'].append("Could not determine base directory")
            return organization_plan
        
        # Create compilation folder
        compilation_folder = os.path.join(base_directory, "compilations")
        
        print(f"\n📁 Organization Plan:")
        print("="*40)
        print(f"Base directory: {base_directory}")
        print(f"Compilation folder: {compilation_folder}")
        
        # Analyze each duplicate group
        for group_idx, group in enumerate(duplicate_analysis['groups'], 1):
            print(f"\n🔍 Group {group_idx}:")
            
            if len(group) < 2:
                continue
                
            # Sort by duration (longest first)
            group.sort(key=lambda x: x['duration'], reverse=True)
            
            # Identify compilation videos (significantly longer than others)
            longest_duration = group[0]['duration']
            compilation_threshold = longest_duration * 0.6  # If others are less than 60% of longest
            
            compilations = []
            individuals = []
            
            for video_info in group:
                if video_info['duration'] >= compilation_threshold:
                    # Check if this video contains sequential matches with shorter ones
                    is_compilation = self._is_compilation_video(
                        video_info, group, duplicate_analysis['duplicates']
                    )
                    
                    if is_compilation:
                        compilations.append(video_info)
                    else:
                        individuals.append(video_info)
                else:
                    individuals.append(video_info)
            
            # Plan moves
            print(f"  📀 Compilations to move: {len(compilations)}")
            for comp in compilations:
                duration_str = f"{comp['duration']//60:.0f}m{comp['duration']%60:.0f}s"
                print(f"    🎬 {comp['filename']} ({duration_str})")
                
                source_path = comp['video']
                filename = os.path.basename(source_path)
                dest_path = os.path.join(compilation_folder, filename)
                
                organization_plan['compilation_moves'].append({
                    'source': source_path,
                    'destination': dest_path,
                    'filename': filename,
                    'duration': comp['duration'],
                    'size_gb': os.path.getsize(source_path) / (1024**3) if os.path.exists(source_path) else 0
                })
            
            print(f"  📄 Individual episodes to keep: {len(individuals)}")
            for ind in individuals:
                duration_str = f"{ind['duration']//60:.0f}m{ind['duration']%60:.0f}s"
                print(f"    📺 {ind['filename']} ({duration_str})")
                organization_plan['individual_keeps'].append(ind)
        
        # Execute moves if requested
        if move_files:
            print(f"\n🚀 Executing file moves...")
            
            # Create compilation directory
            try:
                os.makedirs(compilation_folder, exist_ok=True)
                print(f"✅ Created directory: {compilation_folder}")
            except Exception as e:
                error_msg = f"Failed to create directory {compilation_folder}: {e}"
                print(f"❌ {error_msg}")
                organization_plan['errors'].append(error_msg)
                return organization_plan
            
            # Move compilation files
            moved_count = 0
            for move_plan in organization_plan['compilation_moves']:
                try:
                    source = move_plan['source']
                    destination = move_plan['destination']
                    
                    if os.path.exists(source):
                        # Check if destination already exists
                        if os.path.exists(destination):
                            print(f"⚠️  Destination exists, skipping: {move_plan['filename']}")
                            continue
                        
                        # Move the file
                        import shutil
                        shutil.move(source, destination)
                        print(f"✅ Moved: {move_plan['filename']}")
                        moved_count += 1
                    else:
                        error_msg = f"Source file not found: {source}"
                        print(f"❌ {error_msg}")
                        organization_plan['errors'].append(error_msg)
                        
                except Exception as e:
                    error_msg = f"Failed to move {move_plan['filename']}: {e}"
                    print(f"❌ {error_msg}")
                    organization_plan['errors'].append(error_msg)
            
            print(f"\n📊 Move Summary:")
            print(f"  Files moved: {moved_count}")
            print(f"  Files kept in place: {len(organization_plan['individual_keeps'])}")
            print(f"  Errors: {len(organization_plan['errors'])}")
        
        else:
            print(f"\n📋 Organization Summary (DRY RUN):")
            print(f"  Compilations to move: {len(organization_plan['compilation_moves'])}")
            print(f"  Individual episodes to keep: {len(organization_plan['individual_keeps'])}")
            print(f"  Use move_files=True to execute moves")
        
        return organization_plan
    
    def _is_compilation_video(self, video_info, group, all_duplicates):
        """
        Determine if a video is a compilation by checking for sequential matches.
        
        Args:
            video_info (dict): Video to check
            group (list): All videos in the duplicate group
            all_duplicates (list): All duplicate pair results
            
        Returns:
            bool: True if this appears to be a compilation video
        """
        video_path = video_info['video']
        sequential_matches = 0
        total_comparisons = 0
        
        # Check all duplicate pairs involving this video
        for dup in all_duplicates:
            if dup['video1'] == video_path or dup['video2'] == video_path:
                total_comparisons += 1
                
                # Check if this is a sequential match where current video is longer
                if dup.get('match_type') == 'sequential_duplicate':
                    if dup.get('longer_is_first') and dup['video1'] == video_path:
                        sequential_matches += 1
                    elif not dup.get('longer_is_first') and dup['video2'] == video_path:
                        sequential_matches += 1
        
        # If this video has multiple sequential matches, it's likely a compilation
        return sequential_matches >= 2 or (sequential_matches >= 1 and total_comparisons >= 3)

    def save_fingerprint(self, fingerprint, output_dir='fingerprints'):
        """
        Save fingerprint to JSON file.
        
        Args:
            fingerprint (dict): Fingerprint data
            output_dir (str): Output directory for fingerprint files
            
        Returns:
            str: Path to saved fingerprint file
        """
        if not fingerprint:
            return None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename based on fingerprint code
        fingerprint_code = fingerprint['fingerprint_code']
        filename = f"{fingerprint_code}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Save fingerprint
        try:
            with open(filepath, 'w') as f:
                json.dump(fingerprint, f, indent=2)
            print(f"💾 Fingerprint saved: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ Error saving fingerprint: {e}")
            return None
    
    def load_fingerprint(self, filepath):
        """
        Load fingerprint from JSON file.
        
        Args:
            filepath (str): Path to fingerprint file
            
        Returns:
            dict: Fingerprint data
        """
        try:
            with open(filepath, 'r') as f:
                fingerprint = json.load(f)
            return fingerprint
        except Exception as e:
            print(f"❌ Error loading fingerprint: {e}")
            return None
    
    def process_directory(self, directory_path, save_fingerprints=True):
        """
        Process all videos in a directory and create fingerprints.
        
        Args:
            directory_path (str): Path to directory containing videos
            save_fingerprints (bool): Whether to save fingerprints to files
            
        Returns:
            list: List of fingerprint data
        """
        video_files = []
        for ext in self.supported_formats:
            video_files.extend(Path(directory_path).glob(f"*{ext}"))
            video_files.extend(Path(directory_path).glob(f"*{ext.upper()}"))
        
        if not video_files:
            print(f"❌ No video files found in {directory_path}")
            return []
        
        print(f"🎬 Processing {len(video_files)} video files for fingerprinting...")
        
        fingerprints = []
        for i, video_file in enumerate(video_files, 1):
            print(f"\n--- Processing {i}/{len(video_files)} ---")
            fingerprint = self.extract_fingerprint(str(video_file))
            
            if fingerprint:
                fingerprints.append(fingerprint)
                
                if save_fingerprints:
                    self.save_fingerprint(fingerprint)
            else:
                print(f"❌ Failed to create fingerprint for {video_file.name}")
        
        print(f"\n✅ Created fingerprints for {len(fingerprints)}/{len(video_files)} videos")
        return fingerprints


def main():
    """Test the fingerprinting system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Video Fingerprinting System')
    parser.add_argument('path', help='Video file or directory path')
    parser.add_argument('--save', action='store_true', help='Save fingerprints to files')
    parser.add_argument('--compare', help='Compare with another video file')
    
    args = parser.parse_args()
    
    fingerprinter = VideoFingerprinter()
    
    if os.path.isfile(args.path):
        # Single file
        fingerprint = fingerprinter.extract_fingerprint(args.path)
        if fingerprint:
            # Display title information
            title_info = fingerprinter.get_title_info(fingerprint)
            print(f"\n📺 Title Information:")
            print(f"   Title: {title_info['title']}")
            print(f"   Verified: {'✅ Yes' if title_info['verified'] else '❌ No'}")
            print(f"   Source: {title_info['source']}")
            print(f"   Confidence: {title_info['confidence']:.2f}")
            
            if args.save:
                fingerprinter.save_fingerprint(fingerprint)
        
        # Compare if requested
        if args.compare and fingerprint:
            fp2 = fingerprinter.extract_fingerprint(args.compare)
            if fp2:
                comparison = fingerprinter.compare_fingerprints(fingerprint, fp2)
                print(f"\n🔍 Comparison Results:")
                print(f"   Similar: {comparison['similar']}")
                print(f"   Similarity: {comparison['similarity']:.3f}")
                print(f"   Reason: {comparison['reason']}")
                
                # Show title comparison
                title1 = fingerprinter.get_title_info(fingerprint)
                title2 = fingerprinter.get_title_info(fp2)
                print(f"\n📺 Title Comparison:")
                print(f"   File 1: {title1['title']} ({'verified' if title1['verified'] else 'unverified'})")
                print(f"   File 2: {title2['title']} ({'verified' if title2['verified'] else 'unverified'})")
    
    elif os.path.isdir(args.path):
        # Directory
        fingerprints = fingerprinter.process_directory(args.path, args.save)
        
        # Check for duplicates within the directory
        print(f"\n🔍 Checking for duplicates...")
        duplicates_found = 0
        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                comparison = fingerprinter.compare_fingerprints(fingerprints[i], fingerprints[j])
                if comparison['similar']:
                    duplicates_found += 1
                    title1 = fingerprinter.get_title_info(fingerprints[i])
                    title2 = fingerprinter.get_title_info(fingerprints[j])
                    print(f"   📹 Duplicate: {title1['title']} ≈ {title2['title']}")
                    print(f"      Files: {fingerprints[i]['filename']} ≈ {fingerprints[j]['filename']}")
                    print(f"      Similarity: {comparison['similarity']:.3f}")
        
        if duplicates_found == 0:
            print(f"   ✅ No duplicates found")
        else:
            print(f"   ⚠️  Found {duplicates_found} potential duplicates")
        
        # Show title summary
        print(f"\n📺 Title Summary:")
        verified_count = 0
        for fp in fingerprints:
            title_info = fingerprinter.get_title_info(fp)
            status = "✅" if title_info['verified'] else "❌"
            print(f"   {status} {title_info['title']} ({title_info['source']}, {title_info['confidence']:.2f})")
            if title_info['verified']:
                verified_count += 1
        
        print(f"\n📊 Verification Status: {verified_count}/{len(fingerprints)} titles verified")
    
    else:
        print(f"❌ Path not found: {args.path}")


if __name__ == "__main__":
    main()
