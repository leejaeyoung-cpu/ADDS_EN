"""
Integration Test: CT Analyzer with OpenAI AI Research
Tests the complete workflow from CT analysis to AI research insights
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from medical_imaging.ct_analyzer import CTAnalyzer


def test_ct_analysis_with_ai_integration():
    """
    Complete integration test:
    1. Initialize CTAnalyzer with AI research
    2. Analyze CT image
    3. Get AI research insights
    """
    
    print("=" * 60)
    print("CT Analyzer + OpenAI AI Research Integration Test")
    print("=" * 60)
    
    # Step 1: Initialize analyzer
    print("\n[Step 1] Initializing CTAnalyzer with AI research...")
    analyzer = CTAnalyzer(enable_ai_research=True)
    
    if analyzer.ai_researcher:
        print("[OK] AI Researcher initialized successfully")
        print(f"  - Model: {analyzer.ai_researcher.model}")
    else:
        print("[WARN] AI Researcher not initialized (API key missing)")
        print("  - Set OPENAI_API_KEY in .env to enable AI research")
        return
    
    # Step 2: Analyze CT image (requires actual CT image)
    print("\n[Step 2] CT Image Analysis...")
    
    # Check for test CT images
    test_images = [
        Path("CTdata_cleaned/10035.jpg"),
        Path("CTdata/10035.jpg"),
        Path("test_data/sample_ct.jpg")
    ]
    
    ct_image_path = None
    for img_path in test_images:
        if img_path.exists():
            ct_image_path = str(img_path)
            break
    
    if not ct_image_path:
        print("[WARN] No test CT image found. Skipping analysis.")
        print("  - Place a CT image at: CTdata_cleaned/10035.jpg")
        return
    
    print(f"  - Using: {ct_image_path}")
    
    # Perform analysis with AI research
    print("\n[Step 3] Running analyze_with_ai_research()...")
    results = analyzer.analyze_with_ai_research(
        ct_image_path,
        cancer_type="Colorectal",
        enable_treatment_insights=True
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\nStatus: {results.get('status')}")
    print(f"Modality: {results.get('modality')}")
    
    # Segmentation results
    segmentation = results.get('segmentation', {})
    print(f"\n[Segmentation]")
    print(f"  - Tumor Detected: {segmentation.get('tumor_detected')}")
    print(f"  - Volume: {segmentation.get('tumor_volume_mm3', 0):.2f} mm³")
    
    # Measurements
    measurements = results.get('measurements', {})
    print(f"\n[Measurements]")
    print(f"  - Longest Diameter: {measurements.get('longest_diameter_mm', 0):.2f} mm")
    print(f"  - Shortest Diameter: {measurements.get('shortest_diameter_mm', 0):.2f} mm")
    
    # AI Research Results
    ai_research = results.get('ai_research', {})
    
    if 'analysis' in ai_research:
        print("\n" + "=" * 60)
        print("🤖 AI RESEARCH INSIGHTS")
        print("=" * 60)
        
        print(f"\n[Model Info]")
        print(f"  - Model: {ai_research.get('model', 'N/A')}")
        print(f"  - Tokens Used: {ai_research.get('tokens_used', 0)}")
        print(f"  - Cached: {ai_research.get('cached', False)}")
        
        print(f"\n[Clinical Analysis]")
        print(ai_research['analysis'][:500] + "..." if len(ai_research['analysis']) > 500 else ai_research['analysis'])
        
        if 'tumor_characteristics' in ai_research:
            print(f"\n[Tumor Characteristics Research]")
            print(ai_research['tumor_characteristics'][:500] + "...")
        
        if 'treatment_insights' in ai_research:
            print(f"\n[Treatment Insights]")
            print(ai_research['treatment_insights'][:500] + "...")
        
    else:
        print(f"\n⚠ AI Research Status: {ai_research.get('status', 'unknown')}")
        print(f"  - Message: {ai_research.get('message', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("✅ Integration test completed!")
    print("=" * 60)


def test_individual_research_methods():
    """Test individual AI research methods"""
    
    print("\n" + "=" * 60)
    print("Testing Individual Research Methods")
    print("=" * 60)
    
    analyzer = CTAnalyzer(enable_ai_research=True)
    
    if not analyzer.ai_researcher:
        print("⚠ API key not found, skipping tests")
        return
    
    # Mock CT results
    mock_results = {
        'status': 'success',
        'segmentation': {
            'tumor_detected': True,
            'tumor_volume_mm3': 1500.0
        },
        'measurements': {
            'longest_diameter_mm': 18.5,
            'shortest_diameter_mm': 12.3
        },
        'radiomics_features': {
            'intensity_mean': 45.2,
            'texture_entropy': 2.8,
            'shape_volume_voxels': 850
        }
    }
    
    # Test 1: Research findings
    print("\n[1] Testing research_findings()...")
    research = analyzer.research_findings(mock_results)
    if research:
        print(f"  ✓ Research conducted ({len(research)} chars)")
        print(f"  Preview: {research[:200]}...")
    else:
        print("  ✗ Research failed")
    
    # Test 2: Explain results
    print("\n[2] Testing explain_results()...")
    explanation = analyzer.explain_results(mock_results)
    if explanation:
        print(f"  ✓ Explanation generated ({len(explanation)} chars)")
        print(f"  Preview: {explanation[:200]}...")
    else:
        print("  ✗ Explanation failed")
    
    # Test 3: Suggest treatment
    print("\n[3] Testing suggest_treatment()...")
    treatment = analyzer.suggest_treatment(mock_results)
    if treatment:
        print(f"  ✓ Treatment insights generated ({len(treatment)} chars)")
        print(f"  Preview: {treatment[:200]}...")
    else:
        print("  ✗ Treatment insights failed")
    
    print("\n✅ Individual method tests completed!")


if __name__ == '__main__':
    try:
        test_ct_analysis_with_ai_integration()
        test_individual_research_methods()
        
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
