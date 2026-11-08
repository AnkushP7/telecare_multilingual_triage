import sys
from pathlib import Path

# Fix Windows console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, 'backend')
from services import analyze

cases = [
    'mujhe zor se chest pain ho rha hai aur sar dard bhi',
    '2 din se sar dard aur vomiting lag raha hai, body bahut weak hai.',
    'Since morning chest heavy hai aur saans lene mein thodi dikkat ho rahi hai.',
    'Enakku 3 naal ah gala pain irukku, fever um konjam high ah irukku.'
]

print("================ TELECARE PROTOTYPE (PROJECT REVIEW 2) SMOKE TEST ================")
for text in cases:
    result = analyze(text)
    print('\nINPUT:', text)
    print('LANG:', result['code_mix']['label'])
    print('ENTITIES:', [(e['entity'], e['text'], e.get('snomed_ct', 'N/A'), e.get('icd10', 'N/A')) for e in result['entities']])
    print('TRIAGE:', result['triage']['level'], f"Score: {result['triage']['score']}/100", f"Action: {result['triage']['action']}")
    print('FLAGS:', result['triage'].get('flags', []))
    print('SUMMARY:', result['summary'])
    print('INTAKE REDUCTION:', f"{result['intake_report']['time_metrics']['reduction_percentage']}% faster")
print("\n==================================================================================")
