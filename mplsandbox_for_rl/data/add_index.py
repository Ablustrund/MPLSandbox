import json

ori = json.load(open('/root/StepCoder-main/data/valid.json','r'))

count = 0
data = ori[:16]
for d in data:
    d['id'] = count
    count += 1
    
json.dump(data,open('/root/StepCoder-main/data/valid_2.json','w'),indent=4,ensure_ascii=False)