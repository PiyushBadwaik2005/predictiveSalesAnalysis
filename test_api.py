import urllib.request
import json

def test():
    req = urllib.request.Request('http://127.0.0.1:8000/api/load-sample/ecommerce_retail', data=b'', method='POST')
    with urllib.request.urlopen(req) as resp:
        print('Load sample status:', resp.status)
        data = json.loads(resp.read().decode('utf-8'))
        print('Detected target:', data['inspection']['detected_target'])
        print('Detected features:', data['inspection']['suggested_features'])

    train_payload = json.dumps({
        'target_column': data['inspection']['detected_target'],
        'feature_columns': data['inspection']['suggested_features'],
        'date_column': data['inspection']['detected_date'],
        'sentiment_pct': 10.0,
        'goal_target': 20000.0
    }).encode('utf-8')

    req2 = urllib.request.Request(
        'http://127.0.0.1:8000/api/train-and-predict',
        data=train_payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req2) as resp2:
        print('Train status:', resp2.status)
        res_data = json.loads(resp2.read().decode('utf-8'))
        print('Champion Model:', res_data['champion_model']['name'])
        print('Accuracy R2:', res_data['champion_model']['accuracy_percent'], '%')
        print('Goal Probability:', res_data['goal_analysis']['probability_percent'], '%')
        print('Forecast points:', len(res_data['forecast']['future_projections']))

    sim_payload = json.dumps({
        'custom_inputs': {f: 1000 for f in data['inspection']['suggested_features']},
        'sentiment_pct': 10.0,
        'goal_target': 20000.0
    }).encode('utf-8')

    req3 = urllib.request.Request(
        'http://127.0.0.1:8000/api/simulate-scenario',
        data=sim_payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req3) as resp3:
        print('Scenario simulation status:', resp3.status)
        sim_data = json.loads(resp3.read().decode('utf-8'))
        print('Simulated Sales:', sim_data['predicted_sales'])
        print('Simulated Goal Probability:', sim_data['goal_probability_percent'], '%')

if __name__ == '__main__':
    test()
