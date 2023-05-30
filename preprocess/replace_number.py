'''
replace_number.py
    replace numbers in text with temp_X

'''

text_key = "ori_text"
num_key = "num_list"
dir_path = ""

import json
import os


def read_json_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json_file(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data)

def replace_number(text, num_list: list):
    postfix = ord('a')
    for num in num_list:
        text = text.replace(str(num), f"temp_{chr(postfix)}")
        postfix += 1
    return text

def main():
    print(f"scaning {dir_path}")
    for file in os.listdir(dir_path):
        if file.endswith(".json"):

            print(f"find {file}")
            file_path = os.path.join(dir_path, file)
            data = read_json_file(file_path)

            for datum in data:
                text = datum[text_key]
                num_list = json.loads(datum[num_key])
                new_text = replace_number(text, num_list)
                datum[text_key] = new_text
                datum[num_key] = num_list
                new_data = json.dumps(data, ensure_ascii=False, indent=4)

            new_file_name = os.path.splitext(file)[0] + '_processed' + os.path.splitext(file)[1]
            new_file_path = os.path.join(dir_path, new_file_name)
            print(f"Processed file will be save as {new_file_name}")
            write_json_file(new_file_path, new_data)


if __name__ == "__main__":
    main()