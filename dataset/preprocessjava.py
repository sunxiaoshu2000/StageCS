import jsonlines
import re
import json
import os


def remove_comments_and_docstrings(source):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # 替换为空格而不是空字符串
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    temp = []
    cleaned_source = re.sub(pattern, replacer, source)
    for x in cleaned_source.split('\n'):
        if x.strip() != "":
            temp.append(x)
    return '\n'.join(temp)


if __name__ == "__main__":
    # 只处理Java
    for language in ['java']:
        print(f"Processing {language}")
        train, valid, test = [], [], []

        # 遍历目录收集文件
        for root, dirs, files in os.walk(os.path.join(language, 'final')):
            for file in files:
                file_path = os.path.join(root, file)
                if '.jsonl' in file_path:
                    if 'train' in file_path:
                        train.append(file_path)
                    elif 'valid' in file_path:
                        valid.append(file_path)
                    elif 'test' in file_path:
                        test.append(file_path)

        # 加载数据集
        train_data, valid_data, test_data = {}, {}, {}
        for files, data in [[train, train_data], [valid, valid_data], [test, test_data]]:
            for file in files:
                if '.gz' in file:
                    os.system(f"gzip -d {file}")
                    file = file.replace('.gz', '')
                with open(file) as f:
                    for line in f:
                        js = json.loads(line.strip())
                        data[js['url']] = js

        # 写入处理后的文件
        for tag, data in [['train', train_data], ['valid', valid_data], ['test', test_data]]:
            output_path = os.path.join(language, f"{tag}.jsonl")
            with open(output_path, 'w') as f:
                for url in data:
                    f.write(json.dumps(data[url]) + '\n')

        # 清理代码并保存
        data_types = ['train', 'valid', 'test']
        for mode in data_types:
            cleaned_data = []
            failed_count = 0
            input_path = os.path.join(language, f"{mode}.jsonl")

            with open(input_path, encoding='utf-8') as f:
                for line in f:
                    js = json.loads(line.strip())
                    code = js.get('code', js.get('function', ''))  # 兼容不同字段名

                    try:
                        clean_code = remove_comments_and_docstrings(code)
                    except Exception as e:
                        clean_code = code
                        failed_count += 1

                    js['clean_code'] = clean_code
                    js['clean_doc'] = " ".join(js.get("docstring_tokens", []))
                    cleaned_data.append(js)

            print(f"Processed {len(cleaned_data)} entries ({failed_count} failed) -> {language}/clean_{mode}.jsonl")

            output_path = os.path.join(language, f"clean_{mode}.jsonl")
            with jsonlines.open(output_path, 'w') as w:
                for item in cleaned_data:
                    w.write(item)