#!/usr/bin/env python3
"""
Smart API Integration
Integrates the existing video fingerprinter with the smart search API.
"""
import os
import sys
from pathlib import Path
import json

# Import existing components
try:
    from video_fingerprinter import VideoFingerprinter
    from smart_api_client import SmartFingerprintAPIClient
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure video_fingerprinter.py and smart_api_client.py are in the current directory")
    sys.exit(1)

class SmartVideoAnalyzer:
    """Combines local fingerprinting with smart API search."""
    
    def __init__(self, api_url='http://localhost:5000'):
        self.fingerprinter = VideoFingerprinter()
        self.api_client = SmartFingerprintAPIClient(api_url)
        
    def analyze_and_search(self, video_path, store_if_new=True):
        """
        Analyze a video locally and search for matches using the smart API.
        
        Args:
            video_path (str): Path to the video file
            store_if_new (bool): Whether to store the fingerprint if no matches found
            
        Returns:
            dict: Analysis and search results
        """
        print(f"🎬 Analyzing: {os.path.basename(video_path)}")
        
        # Step 1: Extract fingerprint locally
        try:
            fingerprint_data = self.fingerprinter.extract_fingerprint(video_path)
            if not fingerprint_data:
                return {'error': 'Failed to extract fingerprint'}
                
            print(f"   📄 Fingerprint: {fingerprint_data['fingerprint_code']}")
            
        except Exception as e:
            return {'error': f'Fingerprint extraction failed: {str(e)}'}
        
        # Step 2: Search using smart API
        print(f"   🔍 Searching for matches...")
        search_result = self.api_client.search_fingerprint(
            fingerprint_code=fingerprint_data['fingerprint_code'],
            filename=os.path.basename(video_path)
        )
        
        if 'error' in search_result:
            print(f"   ❌ Search failed: {search_result['error']}")
            return {
                'local_fingerprint': fingerprint_data,
                'search_error': search_result['error']
            }
        
        # Step 3: Analyze results
        matches = search_result.get('results', [])
        best_match = matches[0] if matches else None
        
        print(f"   📊 Found {len(matches)} potential matches")
        
        if best_match and best_match['confidence'] >= 0.8:
            print(f"   ✅ Strong match found: {best_match['confidence']:.3f} confidence")
            print(f"      Type: {best_match['match_type']}")
            print(f"      File: {best_match['match']['filename']}")
            
            result = {
                'local_fingerprint': fingerprint_data,
                'search_results': search_result,
                'best_match': best_match,
                'is_duplicate': True,
                'confidence': best_match['confidence']
            }
            
        elif matches:
            print(f"   ⚠️  Weak matches found (highest: {matches[0]['confidence']:.3f})")
            for match in matches[:3]:  # Show top 3
                print(f"      • {match['match_type']}: {match['confidence']:.3f} - {match['match']['filename']}")
            
            result = {
                'local_fingerprint': fingerprint_data,
                'search_results': search_result,
                'possible_matches': matches,
                'is_duplicate': False,
                'confidence': matches[0]['confidence'] if matches else 0.0
            }
            
        else:
            print(f"   🆕 No matches found - appears to be new content")
            
            result = {
                'local_fingerprint': fingerprint_data,
                'search_results': search_result,
                'is_duplicate': False,
                'is_new': True,
                'confidence': 0.0
            }
        
        # Step 4: Store if new and requested
        if store_if_new and not result.get('is_duplicate', False):
            print(f"   💾 Storing new fingerprint as unverified...")
            
            # Determine media type from filename
            filename = os.path.basename(video_path).lower()
            if any(pattern in filename for pattern in ['s0', 'season', 'episode', 'e0']):
                media_type = 'tv_show'
            elif any(pattern in filename for pattern in ['.19', '.20', 'movie', 'film']):
                media_type = 'movie'
            else:
                media_type = 'unknown'
            
            store_result = self.api_client.store_fingerprint(
                fingerprint_code=fingerprint_data['fingerprint_code'],
                filename=os.path.basename(video_path),
                duration=fingerprint_data.get('duration', 0),
                media_type=media_type,
                segments=fingerprint_data.get('segments', []),
                is_verified=False,  # Store as unverified initially
                verification_source='auto_unverified'
            )
            
            if 'error' not in store_result:
                print(f"   ✅ Stored as unverified fingerprint")
                result['stored'] = True
                result['stored_as'] = 'unverified'
            else:
                print(f"   ❌ Storage failed: {store_result['error']}")
                result['storage_error'] = store_result['error']
        
        return result
    
    def batch_analyze_directory(self, directory_path, file_extensions=None):
        """
        Analyze all videos in a directory.
        
        Args:
            directory_path (str): Path to directory containing videos
            file_extensions (list): List of extensions to process (default: common video formats)
            
        Returns:
            dict: Summary of batch analysis
        """
        if file_extensions is None:
            file_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']
        
        directory = Path(directory_path)
        if not directory.exists():
            return {'error': f'Directory not found: {directory_path}'}
        
        # Find all video files
        video_files = []
        for ext in file_extensions:
            video_files.extend(directory.rglob(f'*{ext}'))
        
        if not video_files:
            return {'error': 'No video files found'}
        
        print(f"🎬 Found {len(video_files)} video files to analyze")
        print("=" * 60)
        
        results = {
            'total_files': len(video_files),
            'processed': 0,
            'duplicates_found': 0,
            'new_content': 0,
            'errors': 0,
            'details': []
        }
        
        for i, video_file in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}] Processing: {video_file.name}")
            
            try:
                analysis = self.analyze_and_search(str(video_file))
                
                if 'error' in analysis:
                    results['errors'] += 1
                    print(f"   ❌ Error: {analysis['error']}")
                else:
                    results['processed'] += 1
                    
                    if analysis.get('is_duplicate', False):
                        results['duplicates_found'] += 1
                    elif analysis.get('is_new', False):
                        results['new_content'] += 1
                
                results['details'].append({
                    'file': str(video_file),
                    'result': analysis
                })
                
            except Exception as e:
                results['errors'] += 1
                results['details'].append({
                    'file': str(video_file),
                    'error': str(e)
                })
                print(f"   ❌ Unexpected error: {str(e)}")
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 BATCH ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total files: {results['total_files']}")
        print(f"Processed successfully: {results['processed']}")
        print(f"Duplicates found: {results['duplicates_found']}")
        print(f"New content: {results['new_content']}")
        print(f"Errors: {results['errors']}")
        
        if results['duplicates_found'] > 0:
            print(f"\n🔍 DUPLICATE DETAILS:")
        for detail in results['details']:
            if detail.get('result', {}).get('is_duplicate', False):
                match = detail['result']['best_match']
                verification_status = "✅ Verified" if match['match']['is_verified'] else "❌ Unverified (Unknown Video)"
                print(f"   • {Path(detail['file']).name}")
                print(f"     Matches: {match['match']['filename']}")
                print(f"     Confidence: {match['confidence']:.3f}")
                print(f"     Type: {match['match_type']}")
                print(f"     Status: {verification_status}")
        
        return results

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Smart Video Analysis with API Search')
    parser.add_argument('path', help='Path to video file or directory')
    parser.add_argument('--api-url', default='http://localhost:5000', 
                       help='API server URL (default: http://localhost:5000)')
    parser.add_argument('--no-store', action='store_true', 
                       help='Don\'t store new fingerprints')
    parser.add_argument('--batch', action='store_true', 
                       help='Process directory in batch mode')
    
    args = parser.parse_args()
    
    analyzer = SmartVideoAnalyzer(args.api_url)
    
    # Check if API is running
    if not analyzer.api_client.health_check():
        print("❌ Smart API is not running!")
        print(f"   Please start the API server: python smart_fingerprint_api.py")
        return 1
    
    print("✅ Connected to Smart API")
    
    if args.batch or os.path.isdir(args.path):
        # Batch processing
        results = analyzer.batch_analyze_directory(args.path)
        if 'error' in results:
            print(f"❌ {results['error']}")
            return 1
    else:
        # Single file processing
        if not os.path.exists(args.path):
            print(f"❌ File not found: {args.path}")
            return 1
        
        result = analyzer.analyze_and_search(args.path, store_if_new=not args.no_store)
        
        if 'error' in result:
            print(f"❌ Analysis failed: {result['error']}")
            return 1
        
        # Print detailed results
        print("\n" + "=" * 50)
        print("📊 ANALYSIS RESULTS")
        print("=" * 50)
        
        if result.get('is_duplicate', False):
            match = result['best_match']
            print(f"🔍 DUPLICATE DETECTED!")
            print(f"   Confidence: {match['confidence']:.3f}")
            print(f"   Match type: {match['match_type']}")
            print(f"   Original file: {match['match']['filename']}")
            print(f"   Reason: {match['reason']}")
        elif result.get('possible_matches', []):
            print(f"⚠️  POSSIBLE MATCHES FOUND:")
            for match in result['possible_matches'][:3]:
                print(f"   • {match['confidence']:.3f} - {match['match']['filename']}")
        else:
            print(f"🆕 NEW CONTENT - No duplicates found")
            if result.get('stored', False):
                print(f"   ✅ Fingerprint stored in database")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
