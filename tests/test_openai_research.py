"""
Tests for OpenAI Medical Research Module
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import module
from src.medical_imaging.ai_research.openai_medical_research import (
    MedicalResearcher,
    ResearchResponse,
    analyze_ct_findings,
    research_tumor_characteristics,
    explain_medical_terms,
    suggest_treatment_insights,
    OPENAI_AVAILABLE
)


class TestMedicalResearcher:
    """Test MedicalResearcher class"""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create mock OpenAI client"""
        with patch('src.medical_imaging.ai_research.openai_medical_research.OpenAI') as mock:
            # Mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Mock AI response"
            mock_response.usage.total_tokens = 100
            
            mock_instance = MagicMock()
            mock_instance.chat.completions.create.return_value = mock_response
            mock.return_value = mock_instance
            
            yield mock
    
    @pytest.fixture
    def researcher(self, mock_openai_client):
        """Create MedicalResearcher instance with mocked OpenAI"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            return MedicalResearcher()
    
    def test_initialization_with_api_key(self, mock_openai_client):
        """Test initialization with API key"""
        if not OPENAI_AVAILABLE:
            pytest.skip("OpenAI package not installed")
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            researcher = MedicalResearcher()
            assert researcher.api_key == 'test_key'
            assert researcher.model == 'gpt-4o'
    
    def test_initialization_without_api_key(self, mock_openai_client):
        """Test initialization fails without API key"""
        if not OPENAI_AVAILABLE:
            pytest.skip("OpenAI package not installed")
        
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key not found"):
                MedicalResearcher()
    
    def test_analyze_ct_findings(self, researcher):
        """Test CT findings analysis"""
        if not OPENAI_AVAILABLE:
            pytest.skip("OpenAI package not installed")
        
        findings = {
            'tumor_count': 2,
            'tumor_volume_mm3': 1250.5,
            'max_diameter_mm': 15.2,
            'location': 'Right colon',
            'confidence_score': 0.92
        }
        
        response = researcher.analyze_ct_findings(findings)
        
        assert isinstance(response, ResearchResponse)
        assert response.content == "Mock AI response"
        assert response.tokens_used == 100
        assert not response.cached
    
    def test_research_tumor_characteristics(self, researcher):
        """Test tumor characteristics research"""
        if not OPENAI_AVAILABLE:
            pytest.skip("OpenAI package not installed")
        
        tumor_data = {
            'type': 'adenocarcinoma',
            'size_mm': 25,
            'shape': 'irregular',
            'density_hu': 45
        }
        
        response = researcher.research_tumor_characteristics(tumor_data)
        
        assert isinstance(response, ResearchResponse)
        assert response.content == "Mock AI response"
    
    def test_explain_medical_terms(self, researcher):
        """Test medical term explanations"""
        if not OPENAI_AVAILABLE:
            pytest.skip("OpenAI package not installed")
        
        terms = ['adenocarcinoma', 'metastasis', 'TNM staging']
        
        response = researcher.explain_medical_terms(terms)
        
        assert isinstance(response, ResearchResponse)
        assert response.content == "Mock AI response"
    
    def test_suggest_treatment_insights(self, researcher):
        """Test treatment insights"""
        if not OPENAI_AVAILABLE:
            pytest.skip("OpenAI package not installed")
        
        patient_data = {
            'tnm_stage': 'T3N1M0',
            'tumor_location': 'sigmoid colon',
            'tumor_size_mm': 35,
            'patient_age': '60-70',
            'comorbidities': ['diabetes', 'hypertension']
        }
        
        response = researcher.suggest_treatment_insights(patient_data)
        
        assert isinstance(response, ResearchResponse)
        assert response.content == "Mock AI response"
    
    def test_caching(self, researcher):
        """Test response caching"""
        if not OPENAI_AVAILABLE:
            pytest.skip("OpenAI package not installed")
        
        findings = {
            'tumor_count': 1,
            'tumor_volume_mm3': 500,
            'max_diameter_mm': 10,
            'location': 'colon',
            'confidence_score': 0.85
        }
        
        # First call - not cached
        response1 = researcher.analyze_ct_findings(findings)
        assert not response1.cached
        
        # Second call with same data - should be cached
        response2 = researcher.analyze_ct_findings(findings)
        assert response2.cached
        assert response2.content == response1.content
    
    def test_cache_stats(self, researcher):
        """Test cache statistics"""
        if not OPENAI_AVAILABLE:
            pytest.skip("OpenAI package not installed")
        
        # Initial stats
        stats = researcher.get_cache_stats()
        initial_count = stats['cached_responses']
        
        # Add to cache
        findings = {'tumor_count': 1}
        researcher.analyze_ct_findings(findings)
        
        # Check stats updated
        stats = researcher.get_cache_stats()
        assert stats['cached_responses'] == initial_count + 1
        assert stats['total_tokens_saved'] > 0
    
    def test_clear_cache(self, researcher):
        """Test cache clearing"""
        if not OPENAI_AVAILABLE:
            pytest.skip("OpenAI package not installed")
        
        # Add to cache
        findings = {'tumor_count': 1}
        researcher.analyze_ct_findings(findings)
        
        # Clear cache
        researcher.clear_cache()
        
        # Verify cache is empty
        stats = researcher.get_cache_stats()
        assert stats['cached_responses'] == 0
        assert stats['total_tokens_saved'] == 0


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @pytest.fixture
    def mock_researcher(self):
        """Mock MedicalResearcher class"""
        with patch('src.medical_imaging.ai_research.openai_medical_research.MedicalResearcher') as mock:
            # Mock instance
            mock_instance = MagicMock()
            mock_response = ResearchResponse(
                content="Mock response",
                model="gpt-4o",
                tokens_used=100,
                timestamp=123.456
            )
            mock_instance.analyze_ct_findings.return_value = mock_response
            mock_instance.research_tumor_characteristics.return_value = mock_response
            mock_instance.explain_medical_terms.return_value = mock_response
            mock_instance.suggest_treatment_insights.return_value = mock_response
            
            mock.return_value = mock_instance
            yield mock
    
    def test_analyze_ct_findings_convenience(self, mock_researcher):
        """Test analyze_ct_findings convenience function"""
        if not OPENAI_AVAILABLE:
            pytest.skip("OpenAI package not installed")
        
        findings = {'tumor_count': 1}
        result = analyze_ct_findings(findings)
        
        assert result == "Mock response"
        mock_researcher.assert_called_once()
    
    def test_research_tumor_characteristics_convenience(self, mock_researcher):
        """Test research_tumor_characteristics convenience function"""
        if not OPENAI_AVAILABLE:
            pytest.skip("OpenAI package not installed")
        
        tumor_data = {'type': 'adenocarcinoma'}
        result = research_tumor_characteristics(tumor_data)
        
        assert result == "Mock response"
    
    def test_explain_medical_terms_convenience(self, mock_researcher):
        """Test explain_medical_terms convenience function"""
        if not OPENAI_AVAILABLE:
            pytest.skip("OpenAI package not installed")
        
        terms = ['adenocarcinoma']
        result = explain_medical_terms(terms)
        
        assert result == "Mock response"
    
    def test_suggest_treatment_insights_convenience(self, mock_researcher):
        """Test suggest_treatment_insights convenience function"""
        if not OPENAI_AVAILABLE:
            pytest.skip("OpenAI package not installed")
        
        patient_data = {'tnm_stage': 'T2N0M0'}
        result = suggest_treatment_insights(patient_data)
        
        assert result == "Mock response"
    
    def test_error_handling(self):
        """Test error handling in convenience functions"""
        if not OPENAI_AVAILABLE:
            pytest.skip("OpenAI package not installed")
        
        with patch('src.medical_imaging.ai_research.openai_medical_research.MedicalResearcher') as mock:
            mock.side_effect = Exception("API Error")
            
            result = analyze_ct_findings({'tumor_count': 1})
            
            assert "AI 분석을 사용할 수 없습니다" in result
            assert "API Error" in result


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_openai_not_available(self):
        """Test behavior when OpenAI package not installed"""
        with patch('src.medical_imaging.ai_research.openai_medical_research.OPENAI_AVAILABLE', False):
            with pytest.raises(ImportError, match="OpenAI package required"):
                MedicalResearcher(api_key="test")
    
    def test_empty_findings(self):
        """Test with empty findings dictionary"""
        if not OPENAI_AVAILABLE:
            pytest.skip("OpenAI package not installed")
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            with patch('src.medical_imaging.ai_research.openai_medical_research.OpenAI'):
                result = analyze_ct_findings({})
                # Should not crash, returns AI response or error message
                assert isinstance(result, str)
    
    def test_custom_model_parameters(self):
        """Test initialization with custom model parameters"""
        if not OPENAI_AVAILABLE:
            pytest.skip("OpenAI package not installed")
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            with patch('src.medical_imaging.ai_research.openai_medical_research.OpenAI'):
                researcher = MedicalResearcher(
                    model="gpt-3.5-turbo",
                    max_tokens=1000,
                    temperature=0.5
                )
                
                assert researcher.model == "gpt-3.5-turbo"
                assert researcher.max_tokens == 1000
                assert researcher.temperature == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
