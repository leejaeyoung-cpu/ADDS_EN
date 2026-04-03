"""
Test CT Analyzer with OpenAI AI Research Integration
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from medical_imaging.ct_analyzer import CTAnalyzer


def test_ct_analyzer_initialization_with_ai_research():
    """Test CTAnalyzer initialization with AI research enabled"""
    analyzer = CTAnalyzer(enable_ai_research=True)
    
    # Check if ai_researcher is initialized
    assert hasattr(analyzer, 'ai_researcher')
    # Note: ai_researcher might be None if OpenAI API key is not set
    

def test_ct_analyzer_initialization_without_ai_research():
    """Test CTAnalyzer initialization with AI research disabled"""
    analyzer = CTAnalyzer(enable_ai_research=False)
    
    assert analyzer.ai_researcher is None


def test_analyze_with_ai_research_no_api_key():
    """Test analyze_with_ai_research when AI researcher is not available"""
    analyzer = CTAnalyzer(enable_ai_research=False)
    
    # Create a mock CT image (would normally be a real file)
    # For now, we'll skip the actual analysis test
    # This test verifies the structure is correct
    
    assert hasattr(analyzer, 'analyze_with_ai_research')


def test_research_methods_exist():
    """Test that all research methods are properly defined"""
    analyzer = CTAnalyzer(enable_ai_research=False)
    
    # Verify all new methods exist
    assert hasattr(analyzer, 'analyze_with_ai_research')
    assert hasattr(analyzer, 'research_findings')
    assert hasattr(analyzer, 'explain_results')
    assert hasattr(analyzer, 'suggest_treatment')


def test_ai_research_graceful_failure():
    """Test that AI research fails gracefully without API key"""
    analyzer = CTAnalyzer(enable_ai_research=False)
    
    mock_ct_results = {
        'status': 'success',
        'segmentation': {'tumor_detected': True},
        'measurements': {},
        'radiomics_features': {}
    }
    
    # These should return None or appropriate messages
    research = analyzer.research_findings(mock_ct_results)
    assert research is None  # No AI researcher
    
    explanation = analyzer.explain_results(mock_ct_results)
    assert explanation is None  # No AI researcher
    
    treatment = analyzer.suggest_treatment(mock_ct_results)
    assert treatment is None  # No AI researcher


if __name__ == '__main__':
    print("Running CT Analyzer AI Research Integration Tests...")
    
    # Run basic tests
    print("\n1. Testing initialization with AI research enabled...")
    test_ct_analyzer_initialization_with_ai_research()
    print("✓ Passed")
    
    print("\n2. Testing initialization with AI research disabled...")
    test_ct_analyzer_initialization_without_ai_research()
    print("✓ Passed")
    
    print("\n3. Testing research methods exist...")
    test_research_methods_exist()
    print("✓ Passed")
    
    print("\n4. Testing graceful failure without API key...")
    test_ai_research_graceful_failure()
    print("✓ Passed")
    
    print("\n✅ All tests passed!")
    print("\n💡 Note: To test actual OpenAI integration, set OPENAI_API_KEY in .env")
