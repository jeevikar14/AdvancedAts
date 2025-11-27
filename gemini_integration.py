import google.generativeai as genai
import os
from dotenv import load_dotenv

class FixedGeminiIntegration:
    def __init__(self, api_key=None):
        load_dotenv()
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model = None
        self.is_working = False
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
               
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                
                # Test the model
                test_response = self.model.generate_content("Hello")
                if test_response and test_response.text:
                    self.is_working = True
                    print("✅ Gemini API configured and working")
                else:
                    print("⚠️ Gemini API responded but may have issues")
                    self.is_working = False
            except Exception as e:
                print(f"⚠️ Gemini API configuration failed: {e}")
                self.is_working = False
        else:
            print("⚠️ No Gemini API key found in environment variables")
            print("   Set GEMINI_API_KEY in your .env file to enable AI features")

    def generate_resume_feedback(self, resume_text, job_description, ats_score, features):
        """Generate comprehensive resume feedback"""
        if not self.is_working or not self.model:
            return self._fallback_feedback(ats_score)
        
        try:
            # Truncate inputs to avoid token limits
            resume_preview = resume_text[:1500] if len(resume_text) > 1500 else resume_text
            jd_preview = job_description[:1500] if len(job_description) > 1500 else job_description
            
            prompt = f"""You are an expert ATS (Applicant Tracking System) consultant and career coach.

**ATS Score: {ats_score:.1f}/100**

**Resume Content (preview):**
{resume_preview}

**Job Description (preview):**
{jd_preview}

Please provide a professional, actionable feedback report with the following structure:

## 🎯 Overall Assessment
Provide a 2-3 sentence summary of how well the resume matches the job description.

## ✅ Key Strengths (2-3 points)
List specific strong points in the resume that align well with the job requirements.

## 🔧 Areas for Improvement (3-4 specific points)
Identify concrete areas where the resume can be enhanced to better match the job description.

## 🔑 Recommended Keywords to Add
List 5-8 specific keywords or skills from the job description that are missing or underrepresented in the resume.

## 📋 Formatting & Structure
Brief assessment of the resume's ATS-friendliness (formatting, sections, clarity).

## 💡 Action Items
3-5 specific, actionable steps the candidate should take to improve their resume.

Keep your feedback concise, professional, and actionable. Focus on improvements that will genuinely increase the ATS score."""
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1500,
                )
            )
            
            if response and response.text:
                return response.text.strip()
            else:
                return self._fallback_feedback(ats_score)
                
        except Exception as e:
            print(f"Gemini API error: {e}")
            return self._fallback_feedback(ats_score)

    def _fallback_feedback(self, ats_score):
        """Provide basic feedback when API is unavailable"""
        feedback = f"""## 📊 ATS Score: {ats_score:.1f}/100

**Note:** Gemini API is not configured. Set your GEMINI_API_KEY in the .env file for detailed AI-powered feedback.

### Quick Assessment:
"""
        if ats_score >= 75:
            feedback += "✅ **Excellent Match** - Your resume is well-optimized for this position. Focus on minor refinements."
        elif ats_score >= 60:
            feedback += "👍 **Good Match** - Strong alignment with requirements. Some optimization will increase your chances."
        elif ats_score >= 40:
            feedback += "🤔 **Moderate Fit** - There's potential, but significant improvements are needed to better align with the job description."
        else:
            feedback += "⚠️ **Needs Improvement** - Your resume requires substantial updates to match this job description."
        
        feedback += """

### General Recommendations:
- Carefully review the job description and incorporate relevant keywords
- Quantify your achievements with specific numbers and percentages
- Use strong action verbs (managed, developed, implemented, led)
- Ensure your resume includes clear sections: Summary, Experience, Education, Skills
- Format your resume simply (avoid tables, graphics, complex formatting)
- Tailor your experience descriptions to highlight relevant skills
"""
        return feedback

    def chat_assistant(self, message, chat_history):
        """Handle chat interactions with context"""
        if not self.is_working or not self.model:
            return "⚠️ Gemini API is not configured. Please set GEMINI_API_KEY in your .env file to use the chat assistant."
        
        try:
            # Build conversation context (last 10 messages for context window management)
            history_context = ""
            recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
            
            for chat in recent_history:
                role = "User" if chat['role'] == 'user' else "CareerGPT"
                history_context += f"{role}: {chat['message']}\n"
            
            prompt = f"""You are CareerGPT, a helpful and knowledgeable AI career assistant. You provide expert advice on:
- Resume writing and optimization
- Job search strategies
- Interview preparation
- Career development
- ATS systems and applicant tracking
- Professional skills development

Be friendly, professional, and provide actionable advice. Keep responses concise (2-4 paragraphs max).

**Conversation History:**
{history_context}

**User's Current Question:** {message}

**Your Response:**"""
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.8,
                    max_output_tokens=800,
                )
            )
            
            if response and response.text:
                return response.text.strip()
            else:
                return "I apologize, but I encountered an issue generating a response. Please try rephrasing your question."
                
        except Exception as e:
            print(f"Chat error: {e}")
            return f"⚠️ I encountered an error: {str(e)}. Please try again or check your API configuration."

    def test_connection(self):
        """Test if Gemini API is working"""
        if not self.api_key:
            return False, "No API key found"
        
        try:
            if self.is_working and self.model:
                test_response = self.model.generate_content("Test")
                if test_response and test_response.text:
                    return True, "API is working"
            return False, "API not responding properly"
        except Exception as e:
            return False, f"Error: {str(e)}"

# Global instance
gemini_client = None

def get_gemini_client():
    """Get or create Gemini client singleton"""
    global gemini_client
    if gemini_client is None:
        gemini_client = FixedGeminiIntegration()
    return gemini_client