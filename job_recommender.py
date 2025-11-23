import os
import requests
from datetime import datetime, timedelta
import streamlit as st

class EnhancedJobRecommender:
    """Job recommendations using SerpAPI with enhanced error handling"""
    
    def __init__(self):
        self.serpapi_key = os.getenv('SERPAPI_KEY')
        self.serpapi_working = False
        
        # Test SerpAPI if key exists
        if self.serpapi_key:
            self.serpapi_working = self._test_serpapi()
            if self.serpapi_working:
                print("✅ SerpAPI configured and working")
            else:
                print("⚠️ SerpAPI key found but connection test failed")
        else:
            print("⚠️ SerpAPI Key missing")
            print("   Set SERPAPI_KEY in your .env file to enable job search")
    
    def _test_serpapi(self):
        """Test SerpAPI connection"""
        try:
            params = {
                'engine': 'google',
                'q': 'test',
                'api_key': self.serpapi_key,
                'num': 1
            }
            response = requests.get('https://serpapi.com/search', params=params, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_job_recommendations(self, skills, experience, location="Remote", limit=5):
        """Get job recommendations from SerpAPI"""
        
        if not self.serpapi_working:
            st.error("❌ SerpAPI is not configured or not working properly.")
            st.info("💡 To enable job search:")
            st.code("1. Get an API key from https://serpapi.com\n2. Add SERPAPI_KEY to your .env file\n3. Restart the application")
            return []

        try:
            return self._get_real_jobs(skills, experience, location, limit)
        except Exception as e:
            st.error(f"❌ Error fetching jobs from SerpAPI: {str(e)}")
            st.info("Please check your SerpAPI key and try again.")
            return []
    
    def _get_real_jobs(self, skills, experience, location, limit):
        """Fetch real jobs from SerpAPI"""
        seniority = self._get_seniority_level(experience)
        primary_skill = skills[0] if skills else "software"
        
        # Build search query
        query = f"{seniority} {primary_skill} developer {location}"
        
        print(f"🔍 Searching jobs: {query}")
        
        params = {
            'engine': 'google_jobs',
            'q': query,
            'hl': 'en',
            'api_key': self.serpapi_key,
            'num': min(limit * 2, 20)  # Request more to filter better matches
        }
        
        try:
            response = requests.get('https://serpapi.com/search', params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for errors in response
            if 'error' in data:
                st.error(f"SerpAPI Error: {data['error']}")
                return []
            
            jobs = data.get('jobs_results', [])
            
            if not jobs:
                st.warning("⚠️ No jobs found matching your criteria via SerpAPI.")
                st.info("💡 Try adjusting your search criteria (location, skills) or check back later.")
                return []
            
            recommendations = []
            for job in jobs[:limit]:
                # Extract job details with safe defaults
                title = job.get('title', 'Position Available')
                company = job.get('company_name', 'Company')
                job_location = job.get('location', location)
                
                # Description
                description = job.get('description', 'No description available')
                if len(description) > 300:
                    description = description[:300] + '...'
                
                # Salary - try multiple fields
                salary = (
                    job.get('detected_extensions', {}).get('salary') or
                    job.get('salary', {}).get('formatted') if isinstance(job.get('salary'), dict) else job.get('salary') or
                    'Competitive salary'
                )
                
                # Apply link
                apply_link = None
                if job.get('apply_options'):
                    apply_link = job['apply_options'][0].get('link')
                elif job.get('share_link'):
                    apply_link = job['share_link']
                elif job.get('related_links') and len(job['related_links']) > 0:
                    apply_link = job['related_links'][0].get('link')
                
                # Posted date
                posted_date = 'Recently'
                if job.get('detected_extensions', {}).get('posted_at'):
                    posted_date = job['detected_extensions']['posted_at']
                elif job.get('posted_at'):
                    posted_date = job['posted_at']
                
                recommendations.append({
                    'title': title,
                    'company': company,
                    'location': job_location,
                    'description': description,
                    'salary': salary,
                    'source_url': apply_link or '#',
                    'is_generated': False,
                    'posted_date': posted_date
                })
            
            print(f"✅ Found {len(recommendations)} jobs from SerpAPI")
            return recommendations
            
        except requests.exceptions.Timeout:
            st.error("❌ Request timeout. SerpAPI is taking too long to respond.")
            return []
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Network error: {str(e)}")
            return []
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            return []
    
    def _get_seniority_level(self, experience):
        """Determine seniority level based on years of experience"""
        if experience >= 10:
            return "Lead"
        elif experience >= 7:
            return "Senior"
        elif experience >= 4:
            return "Mid-Level"
        elif experience >= 1:
            return "Junior"
        else:
            return "Entry-Level"
    
    def get_status(self):
        """Get recommender status"""
        return {
            'api_key_set': bool(self.serpapi_key),
            'working': self.serpapi_working,
            'message': 'Ready' if self.serpapi_working else 'Not configured or not working'
        }