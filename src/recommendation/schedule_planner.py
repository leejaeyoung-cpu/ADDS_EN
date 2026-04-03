"""
Treatment schedule planner
Generates detailed administration schedules for drug regimens
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional


class SchedulePlanner:
    """투여 스케줄 최적화 (IP Module 3)"""
    
    def __init__(self):
        # Drug-specific scheduling templates
        self.drug_schedules = {
            '5-FU': {
                'days': [1, 2],  # D1-D2 continuous infusion
                'infusion_hours': 46,
                'repeat_cycle_weeks': 2
            },
            'Leucovorin': {
                'days': [1, 2],
                'infusion_hours': 2,
                'repeat_cycle_weeks': 2
            },
            'Oxaliplatin': {
                'days': [1],
                'infusion_hours': 2,
                'repeat_cycle_weeks': 2
            },
            'Irinotecan': {
                'days': [1],
                'infusion_hours': 1.5,
                'repeat_cycle_weeks': 2
            },
            'Bevacizumab': {
                'days': [1],
                'infusion_hours': 1.5,
                'repeat_cycle_weeks': 2
            },
            'Cetuximab': {
                'days': [1, 8],  # Weekly dosing
                'infusion_hours': 2,
                'repeat_cycle_weeks': 2
            },
            'Pembrolizumab': {
                'days': [1],
                'infusion_hours': 0.5,
                'repeat_cycle_weeks': 3
            },
            'Capecitabine': {
                'days': list(range(1, 15)),  # D1-D14, PO BID
                'administration': 'PO',
                'frequency': 'BID',
                'repeat_cycle_weeks': 3,
                'rest_days': 7
            }
        }
    
    def generate_schedule(
        self,
        regimen: Dict,
        dosage_plan: Dict,
        start_date: str,
        num_cycles: int = 6,
        patient_holidays: Optional[List[str]] = None
    ) -> Dict:
        """
        상세 투여 스케줄 생성
        
        Args:
            regimen: 약물 조합 정보
            dosage_plan: 용량 계획
            start_date: 시작일 (YYYY-MM-DD)
            num_cycles: 사이클 수
            patient_holidays: 환자 불가능 날짜 리스트
            
        Returns:
            상세 스케줄
        """
        schedule = {
            'regimen_name': regimen.get('name', 'Custom'),
            'start_date': start_date,
            'num_cycles': num_cycles,
            'cycles': []
        }
        
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        # Determine cycle length (max of all drugs)
        cycle_weeks = max(
            self.drug_schedules.get(drug, {}).get('repeat_cycle_weeks', 2)
            for drug in regimen['drugs']
        )
        
        for cycle_num in range(1, num_cycles + 1):
            cycle_schedule = {
                'cycle': cycle_num,
                'start_date': current_date.strftime('%Y-%m-%d'),
                'end_date': (current_date + timedelta(weeks=cycle_weeks, days=-1)).strftime('%Y-%m-%d'),
                'drugs': []
            }
            
            # Schedule each drug
            for drug in regimen['drugs']:
                drug_dosage = next(
                    (d for d in dosage_plan['drugs'] if d['drug_name'] == drug),
                    None
                )
                
                if not drug_dosage:
                    continue
                
                drug_schedule = self._schedule_drug(
                    drug,
                    drug_dosage,
                    current_date,
                    cycle_num,
                    patient_holidays
                )
                
                cycle_schedule['drugs'].append(drug_schedule)
            
            schedule['cycles'].append(cycle_schedule)
            
            # Move to next cycle
            current_date += timedelta(weeks=cycle_weeks)
        
        # Calculate estimated completion
        schedule['estimated_completion'] = current_date.strftime('%Y-%m-%d')
        schedule['total_weeks'] = num_cycles * cycle_weeks
        
        return schedule
    
    def _schedule_drug(
        self,
        drug_name: str,
        dosage: Dict,
        cycle_start: datetime,
        cycle_num: int,
        patient_holidays: Optional[List[str]] = None
    ) -> Dict:
        """개별 약물 스케줄 생성"""
        template = self.drug_schedules.get(drug_name, {
            'days': [1],
            'infusion_hours': 1,
            'repeat_cycle_weeks': 2
        })
        
        administration_dates = []
        
        for day in template['days']:
            admin_date = cycle_start + timedelta(days=day-1)
            
            # Skip if on patient holiday
            if patient_holidays and admin_date.strftime('%Y-%m-%d') in patient_holidays:
                admin_date += timedelta(days=1)  # Reschedule to next day
            
            administration_dates.append({
                'date': admin_date.strftime('%Y-%m-%d'),
                'day_of_cycle': day
            })
        
        drug_schedule = {
            'drug_name': drug_name,
            'dose_mg': dosage['final_dose_mg'],
            'route': dosage['administration_route'],
            'administration_dates': administration_dates
        }
        
        # Add infusion details for IV drugs
        if dosage['administration_route'] == 'IV':
            drug_schedule['infusion_duration_hours'] = template.get('infusion_hours', 1)
            drug_schedule['infusion_type'] = 'Continuous' if template.get('infusion_hours', 0) > 12 else 'Standard'
        
        # Add frequency for PO drugs
        if dosage['administration_route'] == 'PO':
            drug_schedule['frequency'] = template.get('frequency', 'BID')
            drug_schedule['duration_days'] = len(template['days'])
        
        return drug_schedule
    
    def generate_calendar_view(self, schedule: Dict) -> List[Dict]:
        """
        달력 형식의 투여 일정
        
        Returns:
            날짜별 투여 정보 리스트
        """
        calendar_events = []
        
        for cycle in schedule['cycles']:
            for drug_info in cycle['drugs']:
                for admin_date_info in drug_info['administration_dates']:
                    event = {
                        'date': admin_date_info['date'],
                        'cycle': cycle['cycle'],
                        'drug': drug_info['drug_name'],
                        'dose': f"{drug_info['dose_mg']} mg",
                        'route': drug_info['route']
                    }
                    
                    if drug_info['route'] == 'IV':
                        event['duration'] = f"{drug_info['infusion_duration_hours']}h infusion"
                    else:
                        event['frequency'] = drug_info['frequency']
                    
                    calendar_events.append(event)
        
        # Sort by date
        calendar_events.sort(key=lambda x: x['date'])
        
        return calendar_events
    
    def identify_conflicts(self, schedule: Dict) -> List[Dict]:
        """
        스케줄 충돌 식별
        
        Returns:
            충돌 정보 리스트
        """
        conflicts = []
        
        for cycle in schedule['cycles']:
            # Check for same-day multiple long infusions
            date_drug_map = {}
            
            for drug_info in cycle['drugs']:
                for admin_date_info in drug_info['administration_dates']:
                    date = admin_date_info['date']
                    
                    if date not in date_drug_map:
                        date_drug_map[date] = []
                    
                    date_drug_map[date].append(drug_info)
            
            # Check each date
            for date, drugs_on_date in date_drug_map.items():
                total_infusion_hours = sum(
                    d.get('infusion_duration_hours', 0) 
                    for d in drugs_on_date
                )
                
                if total_infusion_hours > 12:
                    conflicts.append({
                        'type': 'long_infusion_day',
                        'date': date,
                        'cycle': cycle['cycle'],
                        'total_hours': total_infusion_hours,
                        'drugs': [d['drug_name'] for d in drugs_on_date],
                        'recommendation': '일부 약물을 다음 날로 분산 권장'
                    })
        
        return conflicts
    
    def generate_patient_instructions(self, schedule: Dict) -> str:
        """환자용 복약 안내"""
        instructions = []
        
        instructions.append(f"=== 치료 일정 안내 ===\n")
        instructions.append(f"치료 시작일: {schedule['start_date']}")
        instructions.append(f"총 사이클 수: {schedule['num_cycles']}")
        instructions.append(f"예상 완료일: {schedule['estimated_completion']}\n")
        
        # First cycle details
        first_cycle = schedule['cycles'][0]
        instructions.append(f"첫 번째 사이클 ({first_cycle['start_date']} ~ {first_cycle['end_date']}):\n")
        
        for drug_info in first_cycle['drugs']:
            instructions.append(f"  • {drug_info['drug_name']}")
            instructions.append(f"    용량: {drug_info['dose_mg']} mg ({drug_info['route']})")
            
            dates_str = ', '.join([d['date'] for d in drug_info['administration_dates']])
            instructions.append(f"    투여일: {dates_str}")
            
            if drug_info['route'] == 'PO':
                instructions.append(f"    복용: {drug_info['frequency']} (하루 {drug_info.get('frequency', 'BID')})")
            else:
                instructions.append(f"    주입 시간: {drug_info.get('infusion_duration_hours', 1)}시간")
            
            instructions.append("")
        
        instructions.append("\n주의사항:")
        instructions.append("- 투여 전 충분한 수분 섭취")
        instructions.append("- 부작용 발생 시 즉시 연락")
        instructions.append("- 정기 혈액 검사 필수")
        
        return '\n'.join(instructions)
