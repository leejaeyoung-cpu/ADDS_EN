"""
API 연결 테스트 스크립트
OpenAI API 키가 올바르게 설정되었는지 확인
"""

from dotenv import load_dotenv
from pathlib import Path
import os
import sys

def test_api_connection():
    """API 연결 및 설정 테스트"""
    
    print("=" * 60)
    print("[SEARCH] ADDS API Connection Diagnostic")
    print("=" * 60)
    
    # Step 1: .env 파일 확인
    print("\n[1/5] Checking .env file...")
    env_path = Path(__file__).parent / '.env'
    
    if env_path.exists():
        print(f"[OK] .env file found: {env_path}")
    else:
        print(f"[ERROR] .env file not found!")
        print(f"   Expected location: {env_path}")
        print(f"\nSolution:")
        print(f"   1. Create .env file in {env_path.parent}")
        print(f"   2. Content: OPENAI_API_KEY=sk-your-api-key-here")
        return False
    
    # Step 2: .env 파일 로드
    print("\n[2/5] Loading environment variables...")
    load_dotenv(dotenv_path=env_path)
    
    # Step 3: API 키 확인
    print("\n[3/5] API 키 확인...")
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("[ERROR] OPENAI_API_KEY 환경 변수 없음!")
        print("\n해결 방법:")
        print("   .env 파일에 다음 형식으로 추가:")
        print("   OPENAI_API_KEY=sk-proj-your-actual-key-here")
        return False
    
    if api_key.startswith('sk-'):
        print(f"[OK] API 키 형식 정상: {api_key[:15]}...")
    else:
        print(f"[WARNING] API 키 형식 의심: {api_key[:15]}...")
        print("   OpenAI API 키는 'sk-'로 시작해야 합니다.")
    
    # Step 4: OpenAI 라이브러리 확인
    print("\n[4/5] OpenAI 라이브러리 확인...")
    try:
        from openai import OpenAI
        print("[OK] OpenAI 라이브러리 로드 성공")
    except ImportError as e:
        print(f"[ERROR] OpenAI 라이브러리 없음: {e}")
        print("\n해결 방법:")
        print("   pip install openai")
        return False
    
    # Step 5: API 연결 테스트
    print("\n[5/5] API 연결 테스트 중...")
    try:
        client = OpenAI(api_key=api_key, timeout=10.0, max_retries=2)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Reply with just 'OK'"}
            ],
            max_tokens=10
        )
        
        result = response.choices[0].message.content
        print(f"[OK] API 연결 성공!")
        print(f"   응답: {result}")
        print(f"   모델: {response.model}")
        print(f"   사용 토큰: {response.usage.total_tokens}")
        
        return True
    
    except Exception as e:
        error_type = type(e).__name__
        print(f"[ERROR] API 연결 실패: {error_type}")
        print(f"   상세: {str(e)}")
        
        # 구체적인 에러 진단
        if "authentication" in str(e).lower() or "api_key" in str(e).lower():
            print("\n[TIP] 원인: API 키가 유효하지 않음")
            print("   해결: OpenAI 플랫폼에서 새 API 키 생성")
            print("   https://platform.openai.com/api-keys")
        
        elif "timeout" in str(e).lower():
            print("\n[TIP] 원인: 네트워크 타임아웃")
            print("   해결: 인터넷 연결 확인 또는 방화벽 설정")
        
        elif "rate" in str(e).lower() or "429" in str(e):
            print("\n[TIP] 원인: API 요청 제한 초과")
            print("   해결: 잠시 후 재시도 또는 플랜 업그레이드")
        
        elif "quota" in str(e).lower() or "billing" in str(e).lower():
            print("\n[TIP] 원인: API 크레딧 부족")
            print("   해결: OpenAI 계정에 결제 수단 등록")
        
        else:
            print("\n[TIP] 일반적인 해결 방법:")
            print("   1. 인터넷 연결 확인")
            print("   2. API 키 재확인")
            print("   3. 방화벽/프록시 설정 확인")
        
        return False
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("\n")
    success = test_api_connection()
    
    if success:
        print("\n" + "=" * 60)
        print("[SUCCESS] 모든 테스트 통과!")
        print("=" * 60)
        print("\nADDS 시스템이 OpenAI API와 정상적으로 통신할 수 있습니다.")
        print("Streamlit 앱을 실행하면 AI 기능을 사용할 수 있습니다.")
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("[ERROR] 테스트 실패")
        print("=" * 60)
        print("\n위의 해결 방법을 참고하여 문제를 해결하세요.")
        print("문제가 지속되면 상세 가이드를 확인하세요:")
        print("  → api_connection_troubleshooting.md")
        sys.exit(1)
