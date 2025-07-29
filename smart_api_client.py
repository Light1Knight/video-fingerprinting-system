#!/usr/bin/env python3
"""
Smart Fingerprint API Client
Client for interacting with the smart video fingerprint search API.
"""
import requests
import json
from pathlib import Path

class SmartFingerprintAPIClient:
    def __init__(self, base_url='http://localhost:5000'):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def search_fingerprint(self, fingerprint_code, filename=None):
        """
        Smart search for fingerprint matches.
        
        Args:
            fingerprint_code (str): The fingerprint to search for
            filename (str, optional): Original filename for name extraction
            
        Returns:
            dict: Search results with matches and confidence scores
        """
        url = f"{self.base_url}/api/search"
        data = {
            'fingerprint_code': fingerprint_code
        }
        
        if filename:
            data['filename'] = filename
        
        try:
            response = self.session.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {'error': f'Request failed: {str(e)}'}
    
    def store_fingerprint(self, fingerprint_code, filename, duration=0, 
                         title=None, season=None, episode=None, 
                         release_date=None, media_type='unknown', segments=None,
                         is_verified=False, verification_source=None):
        """
        Store a new fingerprint in the database.
        
        Args:
            fingerprint_code (str): The fingerprint code
            filename (str): Original filename
            duration (float): Video duration in seconds
            title (str, optional): Clean title
            season (int, optional): Season number for TV shows
            episode (int, optional): Episode number for TV shows
            release_date (str, optional): Release date
            media_type (str): 'movie', 'tv_show', or 'unknown'
            segments (list, optional): Segment data
            is_verified (bool): Whether this fingerprint is verified
            verification_source (str, optional): Source of verification
            
        Returns:
            dict: Storage result
        """
        url = f"{self.base_url}/api/fingerprints"
        data = {
            'fingerprint_code': fingerprint_code,
            'filename': filename,
            'duration': duration,
            'media_type': media_type,
            'is_verified': is_verified
        }
        
        if verification_source:
            data['verification_source'] = verification_source
        
        if title:
            data['title'] = title
        if season is not None:
            data['season'] = season
        if episode is not None:
            data['episode'] = episode
        if release_date:
            data['release_date'] = release_date
        if segments:
            data['segments'] = segments
        
        try:
            response = self.session.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {'error': f'Request failed: {str(e)}'}
    
    def verify_fingerprint(self, fingerprint_code, title=None, season=None, 
                          episode=None, release_date=None, media_type=None,
                          verification_source='manual'):
        """
        Mark a fingerprint as verified and add metadata.
        
        Args:
            fingerprint_code (str): The fingerprint code to verify
            title (str, optional): Clean title
            season (int, optional): Season number for TV shows
            episode (int, optional): Episode number for TV shows
            release_date (str, optional): Release date
            media_type (str, optional): 'movie', 'tv_show', or 'unknown'
            verification_source (str): Source of verification
            
        Returns:
            dict: Verification result
        """
        url = f"{self.base_url}/api/fingerprints/{fingerprint_code}/verify"
        data = {
            'verification_source': verification_source
        }
        
        if title is not None:
            data['title'] = title
        if season is not None:
            data['season'] = season
        if episode is not None:
            data['episode'] = episode
        if release_date:
            data['release_date'] = release_date
        if media_type:
            data['media_type'] = media_type
        
        try:
            response = self.session.put(url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {'error': f'Request failed: {str(e)}'}
    
    def add_verified_fingerprint(self, fingerprint_code, filename, duration=0,
                                title=None, season=None, episode=None, 
                                release_date=None, media_type='unknown', 
                                segments=None, verification_source='manual'):
        """
        Smart add verified fingerprint with conflict detection and cleanup.
        
        Args:
            fingerprint_code (str): The fingerprint code
            filename (str): Original filename
            duration (float): Video duration in seconds
            title (str, optional): Clean title
            season (int, optional): Season number for TV shows
            episode (int, optional): Episode number for TV shows
            release_date (str, optional): Release date
            media_type (str): 'movie', 'tv_show', or 'unknown'
            segments (list, optional): Segment data
            verification_source (str): Source of verification
            
        Returns:
            dict: Result with potential conflicts or success
        """
        url = f"{self.base_url}/api/fingerprints/add-verified"
        data = {
            'fingerprint_code': fingerprint_code,
            'filename': filename,
            'duration': duration,
            'media_type': media_type,
            'verification_source': verification_source
        }
        
        if title:
            data['title'] = title
        if season is not None:
            data['season'] = season
        if episode is not None:
            data['episode'] = episode
        if release_date:
            data['release_date'] = release_date
        if segments:
            data['segments'] = segments
        
        try:
            response = self.session.post(url, json=data)
            return response.json()  # Don't raise for 409 conflicts
        except requests.RequestException as e:
            return {'error': f'Request failed: {str(e)}'}
    
    def confirm_verified_fingerprint(self, fingerprint_code, action, 
                                   overwrite_fingerprint_id=None, **kwargs):
        """
        Confirm adding a verified fingerprint after conflict resolution.
        
        Args:
            fingerprint_code (str): The fingerprint code
            action (str): 'overwrite', 'proceed_anyway', or 'cancel'
            overwrite_fingerprint_id (int, optional): ID of fingerprint to overwrite
            **kwargs: Additional data from original request
            
        Returns:
            dict: Final result
        """
        url = f"{self.base_url}/api/fingerprints/add-verified/confirm"
        data = {
            'fingerprint_code': fingerprint_code,
            'action': action,
            **kwargs
        }
        
        if overwrite_fingerprint_id:
            data['overwrite_fingerprint_id'] = overwrite_fingerprint_id
        
        try:
            response = self.session.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {'error': f'Request failed: {str(e)}'}
    
    def get_unverified_fingerprints(self, page=1, per_page=10):
        """
        Get unverified fingerprints for manual verification.
        
        Args:
            page (int): Page number
            per_page (int): Items per page
            
        Returns:
            dict: List of unverified fingerprints
        """
        url = f"{self.base_url}/api/fingerprints/unverified"
        params = {'page': page, 'per_page': per_page}
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {'error': f'Request failed: {str(e)}'}
    
    def get_stats(self):
        """Get database statistics."""
        url = f"{self.base_url}/api/stats"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {'error': f'Request failed: {str(e)}'}
    
    def health_check(self):
        """Check if API is running."""
        try:
            response = self.session.get(f"{self.base_url}/")
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False

def demonstrate_smart_search():
    """Demonstrate the smart search functionality."""
    client = SmartFingerprintAPIClient()
    
    print("🔍 Smart Fingerprint Search Demo")
    print("=" * 50)
    
    # Check if API is running
    if not client.health_check():
        print("❌ API is not running. Please start the API server first.")
        print("   Run: python smart_fingerprint_api.py")
        return
    
    print("✅ API is running!")
    
    # Get current stats
    stats = client.get_stats()
    print(f"\n📊 Current Database Stats:")
    print(f"   Total fingerprints: {stats.get('total_fingerprints', 0)}")
    print(f"   Verified: {stats.get('verified_fingerprints', 0)}")
    print(f"   Unverified: {stats.get('unverified_fingerprints', 0)}")
    print(f"   Movies: {stats.get('movies', 0)}")
    print(f"   TV Shows: {stats.get('tv_shows', 0)}")
    
    # Example searches - mix of verified and unverified
    test_cases = [
        {
            'fingerprint': 'ABCD1234EFGH5678IJKL9012MNOP3456-TEST',
            'filename': 'Breaking.Bad.S03E07.One.Minute.avi',
            'description': 'TV show episode (will be verified)',
            'verify': True,
            'verify_data': {
                'title': 'Breaking Bad',
                'season': 3,
                'episode': 7,
                'media_type': 'tv_show',
                'release_date': '2010-05-02'
            }
        },
        {
            'fingerprint': 'UNKNOWN123456789MYSTERY987654321ABC-UNK',
            'filename': 'Unknown.Video.File.mp4',
            'description': 'Unknown video (will remain unverified)',
            'verify': False
        },
        {
            'fingerprint': 'FEDEEEDCBDDCDCABBBBA9AAA12345678-MOVIE',
            'filename': 'The.Matrix.1999.720p.BluRay.x264.mp4',
            'description': 'Movie (will be verified)',
            'verify': True,
            'verify_data': {
                'title': 'The Matrix',
                'media_type': 'movie',
                'release_date': '1999-03-31'
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}: {test_case['description']}")
        print(f"   Fingerprint: {test_case['fingerprint']}")
        print(f"   Filename: {test_case['filename']}")
        
        # First store it (unverified by default)
        store_result = client.store_fingerprint(
            fingerprint_code=test_case['fingerprint'],
            filename=test_case['filename'],
            duration=2700,  # 45 minutes
            media_type='tv_show' if 'S0' in test_case['filename'] else 'movie'
        )
        
        if 'error' not in store_result:
            print(f"   ✅ Stored successfully (unverified)")
        else:
            print(f"   ℹ️  {store_result.get('message', 'Already exists')}")
        
        # Verify if requested
        if test_case.get('verify', False) and 'verify_data' in test_case:
            verify_result = client.verify_fingerprint(
                fingerprint_code=test_case['fingerprint'],
                **test_case['verify_data']
            )
            
            if 'error' not in verify_result:
                print(f"   ✅ Verified with metadata")
            else:
                print(f"   ❌ Verification failed: {verify_result['error']}")
        
        # Then search for it
        search_result = client.search_fingerprint(
            fingerprint_code=test_case['fingerprint'],
            filename=test_case['filename']
        )
        
        if 'error' in search_result:
            print(f"   ❌ Search failed: {search_result['error']}")
            continue
        
        print(f"   📊 Found {search_result['matches_found']} matches:")
        
        for match in search_result['results']:
            print(f"      • {match['match_type']}: {match['confidence']:.3f} confidence")
            print(f"        {match['match']['filename']}")
            print(f"        Verified: {'✅' if match['match']['is_verified'] else '❌'}")
            print(f"        Reason: {match['reason']}")
    
    # Test searching for a completely new video (should be saved as unverified)
    print(f"\n🆕 Testing New Unknown Video:")
    new_search = client.search_fingerprint(
        fingerprint_code='NEWVIDEO123456789UNKNOWN987654321NEW-UNK',
        filename='Completely.Unknown.Video.2024.mp4'
    )
    
    if 'action_taken' in new_search and new_search['action_taken'] == 'saved_as_unverified':
        print(f"   ✅ New video saved as unverified fingerprint")
        print(f"   📄 Fingerprint: {new_search['unverified_fingerprint']['fingerprint_code']}")
    
    # Show unverified fingerprints
    print(f"\n📋 Unverified Fingerprints:")
    unverified = client.get_unverified_fingerprints()
    if 'error' not in unverified:
        for fp in unverified['unverified_fingerprints'][:3]:  # Show first 3
            print(f"   • {fp['filename']} ({fp['fingerprint_code'][:16]}...)")
            print(f"     Created: {fp['created_at'][:19]}")
    
    # Demonstrate partial matching
    print(f"\n🔍 Testing Partial Matching:")
    partial_search = client.search_fingerprint(
        fingerprint_code='ABCD1234DIFFERENT789IJKL9012MNOP3456-TEST',
        filename='Breaking.Bad.S03E08.Different.Episode.avi'
    )
    
    if 'error' not in partial_search:
        print(f"   📊 Found {partial_search['matches_found']} matches for similar fingerprint:")
        for match in partial_search['results']:
            verification_status = "verified" if match['match']['is_verified'] else "unverified"
            print(f"      • {match['match_type']}: {match['confidence']:.3f} confidence ({verification_status})")
    
    # Final stats
    final_stats = client.get_stats()
    print(f"\n📊 Final Database Stats:")
    print(f"   Total fingerprints: {final_stats.get('total_fingerprints', 0)}")
    print(f"   Verified: {final_stats.get('verified_fingerprints', 0)}")
    print(f"   Unverified: {final_stats.get('unverified_fingerprints', 0)}")
    print(f"   Movies: {final_stats.get('movies', 0)}")
    print(f"   TV Shows: {final_stats.get('tv_shows', 0)}")
    
    # Demonstrate add verified fingerprint functionality
    print(f"\n🔐 Testing Add Verified Fingerprint:")
    
    # Test 1: Add a completely new verified fingerprint
    print(f"Test 1: Adding new verified fingerprint")
    add_result1 = client.add_verified_fingerprint(
        fingerprint_code='VERIFIED123456789NEW987654321VER-NEW',
        filename='New.Verified.Show.S01E01.mp4',
        title='New Verified Show',
        season=1,
        episode=1,
        media_type='tv_show',
        duration=2400
    )
    
    if add_result1.get('result') == 'success':
        print(f"   ✅ Successfully added new verified fingerprint")
        print(f"   🧹 Cleaned up {add_result1.get('cleanup_count', 0)} unverified matches")
    elif 'error' in add_result1:
        print(f"   ❌ Error: {add_result1['error']}")
    
    # Test 2: Try to add a fingerprint that might conflict
    print(f"\nTest 2: Testing conflict detection")
    add_result2 = client.add_verified_fingerprint(
        fingerprint_code='CONFLICT123456789TEST987654321CON-FLT',
        filename='Breaking.Bad.S03E07.Similar.Name.mp4',  # Similar to existing
        title='Breaking Bad',
        season=3,
        episode=7,
        media_type='tv_show',
        duration=2700
    )
    
    if add_result2.get('result') == 'conflict_detected':
        print(f"   ⚠️  Conflict detected!")
        print(f"   📋 Action required: {add_result2.get('action_required')}")
        print(f"   📊 Name similarity: {add_result2.get('conflict_details', {}).get('name_similarity', 0)}")
        
        # Demonstrate resolving the conflict
        print(f"   🔧 Resolving conflict by proceeding anyway...")
        confirm_result = client.confirm_verified_fingerprint(
            fingerprint_code='CONFLICT123456789TEST987654321CON-FLT',
            action='proceed_anyway',
            filename='Breaking.Bad.S03E07.Similar.Name.mp4',
            title='Breaking Bad',
            season=3,
            episode=7,
            media_type='tv_show',
            duration=2700
        )
        
        if confirm_result.get('result') == 'success':
            print(f"   ✅ Conflict resolved - fingerprint added")
        else:
            print(f"   ❌ Failed to resolve conflict: {confirm_result.get('error', 'Unknown error')}")
    
    elif add_result2.get('result') == 'success':
        print(f"   ✅ No conflicts - fingerprint added successfully")
    else:
        print(f"   ℹ️  Result: {add_result2.get('message', 'Unknown result')}")

def interactive_add_verified():
    """Interactive function to add verified fingerprints with conflict handling."""
    client = SmartFingerprintAPIClient()
    
    if not client.health_check():
        print("❌ API is not running. Please start the API server first.")
        return
    
    print("🔐 Interactive Add Verified Fingerprint")
    print("=" * 50)
    
    # Get input from user (in a real implementation)
    fingerprint_code = input("Enter fingerprint code: ").strip()
    filename = input("Enter filename: ").strip()
    title = input("Enter title (optional): ").strip() or None
    
    # Try to add the verified fingerprint
    result = client.add_verified_fingerprint(
        fingerprint_code=fingerprint_code,
        filename=filename,
        title=title,
        duration=2700,  # Default duration
        verification_source='manual'
    )
    
    # Handle the result
    if result.get('result') == 'success':
        print(f"✅ Successfully added verified fingerprint!")
        if result.get('cleanup_count', 0) > 0:
            print(f"🧹 Automatically cleaned up {result['cleanup_count']} unverified duplicates")
    
    elif result.get('result') == 'conflict_detected':
        print(f"⚠️  Conflict detected with existing fingerprint!")
        print(f"📋 Existing: {result['existing_match']['filename']}")
        print(f"📊 Name similarity: {result['conflict_details']['name_similarity']}")
        
        choice = input("Choose action (overwrite/proceed/cancel): ").strip().lower()
        
        if choice in ['overwrite', 'proceed_anyway', 'cancel']:
            confirm_result = client.confirm_verified_fingerprint(
                fingerprint_code=fingerprint_code,
                action=choice,
                overwrite_fingerprint_id=result['existing_match']['id'] if choice == 'overwrite' else None,
                filename=filename,
                title=title,
                duration=2700,
                verification_source='manual'
            )
            
            if confirm_result.get('result') == 'success':
                print(f"✅ Action completed successfully!")
            else:
                print(f"❌ Action failed: {confirm_result.get('error', 'Unknown error')}")
        else:
            print("❌ Invalid choice. Operation cancelled.")
    
    elif result.get('result') == 'already_exists_verified':
        print(f"ℹ️  Fingerprint already exists and is verified")
    
    elif result.get('result') == 'upgraded_from_unverified':
        print(f"✅ Existing unverified fingerprint upgraded to verified!")
    
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    demonstrate_smart_search()
    
    # Uncomment to run interactive mode
    # interactive_add_verified()
