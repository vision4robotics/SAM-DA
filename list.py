import json
import re

# 打开 JSON 文件
char_at_position = []
for i in range(4):
    with open('./result/'+str(i)+'.json') as file:
        # 读取整个文件内容
        content = file.read()
        pattern = r'\}(\r?\n)\{'
        content = re.sub(pattern,'};\n{',content)
        dictionaries = content.split(';')
    for dictionary in dictionaries:
        data = json.loads(dictionary)
        with open('./result/list.json', 'a') as file:
                json.dump(data, file, indent=4, sort_keys=True)
                file.write('\n')