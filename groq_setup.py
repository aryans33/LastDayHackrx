#!/usr/bin/env python3
"""
GROQ Setup Instructions for HackRX Document Q&A System
"""

def show_groq_setup():
    """Display GROQ setup instructions"""
    print("🚀 GROQ Setup Instructions")
    print("=" * 50)
    
    print("\n📋 Step 1: Get GROQ API Key")
    print("-" * 30)
    print("1. Go to: https://console.groq.com/")
    print("2. Sign up for a free account")
    print("3. Navigate to 'API Keys' section")
    print("4. Create a new API key")
    print("5. Copy the generated key")
    
    print("\n📋 Step 2: Update .env File")
    print("-" * 30)
    print("Edit your .env file and replace:")
    print("GROQ_API_KEY=your_groq_api_key_here")
    print("with your actual GROQ API key")
    
    print("\n📋 Step 3: Why GROQ?")
    print("-" * 25)
    print("✅ Much faster than local Ollama")
    print("✅ More reliable than local setup")
    print("✅ Free tier with good limits")
    print("✅ Uses Llama-3.1-70B model (very capable)")
    print("✅ OpenAI-compatible API")
    
    print("\n📋 Step 4: Test Configuration")
    print("-" * 35)
    print("After updating your API key:")
    print("1. Restart your FastAPI server")
    print("2. Run: python test_system.py")
    print("3. You should see actual answers instead of errors")
    
    print("\n📋 Step 5: Expected Results")
    print("-" * 30)
    print("With GROQ, you should get answers like:")
    print("- Grace period: 'A grace period of thirty days is provided...'")
    print("- Maternity coverage: 'Yes, the policy covers maternity expenses...'")
    print("- Cataract surgery: 'The policy has a waiting period of two years...'")
    
    print("\n🎯 Benefits of GROQ over Local Mistral:")
    print("⚡ 10x faster response times")
    print("🔧 No local setup required")
    print("💾 No memory/disk usage")
    print("🌐 Always available")
    print("📊 Better JSON formatting")
    
    print("\n🔧 Troubleshooting:")
    print("❌ API key errors: Verify key is correct in .env")
    print("❌ Rate limits: GROQ has generous free tier")
    print("❌ Network issues: Check internet connection")
    
    print("\n🚀 Ready to test? Run:")
    print("python test_system.py")

if __name__ == "__main__":
    show_groq_setup()
