import yaml
with open("data/lipad_crack.yaml", 'r') as f:
    data = yaml.safe_load(f)
    print(f"File Content: {data}")
    print(f"Type of Data: {type(data)}")