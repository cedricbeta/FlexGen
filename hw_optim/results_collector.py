import json
import os

def view():
    data = []
    with open('data.json', 'r') as f:
        data = json.load(f)
    return data

def store_entry(entry):
    data = []
    with open('data.json', 'r') as f:
        data = json.load(f)
    data.append(entry)
    with open('data.json', 'w') as f:
        json.dump(data, f)

def overwrite(data):
    with open('data.json', 'w') as f:
        json.dump(data, f)
        
def main():
    data = view()
    table_keys = []
    table_entries = []
    for entry in data:
        for k in entry.keys():
            if k not in table_keys:
                table_keys.append(k)
    for entry in data:
        table_entry = []
        for tk in table_keys:
            if tk not in entry.keys():
                table_entry.append(None)
            else:
                table_entry.append(entry[tk])
        table_entries.append(table_entry)
    print('{','c'*len(table_keys),'}\n\\hline\n', ' & '.join(map(str, table_keys)), '\\\\\n\\hline')
    for te in table_entries:
        print(' & '.join(map(str, te)), '\\\\\n\\hline')
    return

if __name__ == '__main__':
    main()