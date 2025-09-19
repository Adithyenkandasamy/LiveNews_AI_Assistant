#!/usr/bin/env python3
"""
Test script to validate Gemini migration
Runs without requiring API key for basic validation
"""

import sys
import traceback
from datetime import datetime

def test_imports():
    """Test that all modules can be imported"""
    print("=== Testing Imports ===")
    try:
        from gemini_client import GeminiClient
        print("✅ gemini_client imported successfully")
        
        from app import LiveNewsAI
        print("✅ app module imported successfully")
        
        from pathway_rag_system import PathwayRAGSystem, PathwayRAGConfig
        print("✅ pathway_rag_system imported successfully")
        
        from news_freshness_validator import NewsFreshnessValidator, FreshnessConfig
        print("✅ news_freshness_validator imported successfully")
        
        from enhanced_summarizer import EnhancedSummarizer
        print("✅ enhanced_summarizer imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_gemini_client_structure():
    """Test Gemini client structure without API calls"""
    print("\n=== Testing Gemini Client Structure ===")
    try:
        from gemini_client import GeminiClient
        
        # Test initialization without API key
        client = GeminiClient()
        print("✅ GeminiClient can be instantiated")
        
        # Check required methods exist
        assert hasattr(client, 'generate_response')
        print("✅ generate_response method exists")
        
        assert hasattr(client, 'test_connection')
        print("✅ test_connection method exists")
        
        assert hasattr(client, 'available')
        print("✅ available property exists")
        
        # Test model info
        info = client.get_model_info()
        assert isinstance(info, dict)
        print("✅ get_model_info returns dict")
        
        return True
    except Exception as e:
        print(f"❌ Gemini client test error: {e}")
        traceback.print_exc()
        return False

def test_config_structures():
    """Test configuration classes"""
    print("\n=== Testing Configuration Structures ===")
    try:
        from pathway_rag_system import PathwayRAGConfig
        from news_freshness_validator import FreshnessConfig
        from gemini_client import GeminiClient
        
        # Test PathwayRAGConfig
        client = GeminiClient()
        config = PathwayRAGConfig(gemini_client=client)
        assert config.gemini_client is not None
        print("✅ PathwayRAGConfig accepts gemini_client")
        
        # Test FreshnessConfig
        fresh_config = FreshnessConfig(gemini_client=client)
        assert fresh_config.gemini_client is not None
        print("✅ FreshnessConfig accepts gemini_client")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test error: {e}")
        traceback.print_exc()
        return False

def test_enhanced_summarizer():
    """Test Enhanced Summarizer with different models"""
    print("\n=== Testing Enhanced Summarizer ===")
    try:
        from enhanced_summarizer import EnhancedSummarizer
        from gemini_client import GeminiClient
        
        # Test Gemini mode
        client = GeminiClient()
        summarizer = EnhancedSummarizer("gemini", client)
        assert summarizer.model_type == "gemini"
        assert summarizer.gemini_client is not None
        print("✅ EnhancedSummarizer works with Gemini")
        
        # Test Ollama mode (backward compatibility)
        ollama_summarizer = EnhancedSummarizer("ollama", "llama3.2:3b")
        assert ollama_summarizer.model_type == "ollama"
        assert ollama_summarizer.model_name == "llama3.2:3b"
        print("✅ EnhancedSummarizer maintains Ollama compatibility")
        
        return True
    except Exception as e:
        print(f"❌ Enhanced Summarizer test error: {e}")
        traceback.print_exc()
        return False

def test_app_initialization():
    """Test that main app can initialize"""
    print("\n=== Testing App Initialization ===")
    try:
        from app import LiveNewsAI
        
        # This will initialize without API key but should not crash
        print("⏳ Initializing LiveNewsAI (this may take a moment)...")
        news_ai = LiveNewsAI()
        
        assert hasattr(news_ai, 'gemini_client')
        print("✅ LiveNewsAI has gemini_client attribute")
        
        assert hasattr(news_ai, 'model_available')
        print("✅ LiveNewsAI has model_available attribute")
        
        # Test news feed configuration
        assert len(news_ai.news_feeds) > 0
        print(f"✅ LiveNewsAI loaded {len(news_ai.news_feeds)} news sources")
        
        return True
    except Exception as e:
        print(f"❌ App initialization error: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all migration tests"""
    print(f"🚀 Starting Gemini Migration Validation")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_gemini_client_structure,
        test_config_structures,
        test_enhanced_summarizer,
        test_app_initialization
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Gemini migration appears successful.")
        print("\nNext steps:")
        print("1. Set your GEMINI_API_KEY in .env file")
        print("2. Run: python app.py")
        print("3. Visit http://localhost:5000")
    else:
        print("⚠️  Some tests failed. Check errors above.")
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
