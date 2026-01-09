import pandas as pd

# 读取CSV文件
file_path = 'E:/D/数据分析.csv'
df = pd.read_csv(file_path)

# 假设'periodontitis'和'jawbone loss'是列名
# 根据'periodontitis'列的值来更新'jawbone loss'列
# 如果'periodontitis'为0，则'jawbone loss'也设为0
# 如果'periodontitis'为1，则'jawbone loss'设为1
# 注意：这里使用了条件表达式来实现
df['jawbone loss'] = df['periodontitis'].apply(lambda x: 0 if x == 0 else 1)

# 如果你想要一个名为'jawbinloss'的新列，而不是更新'jawbone loss'列，
# 你可以这样做（但请注意，这将保留原始的'jawbone loss'列）：
# df['jawbinloss'] = df['periodontitis'].apply(lambda x: 0 if x == 0 else 1)

# 写回CSV文件，可以选择不覆盖原文件
output_file_path = 'E:/D/更新后的数据分析.csv'  # 更改文件名以避免覆盖原始文件
df.to_csv(output_file_path, index=False)  # index=False表示不将行索引写入文件

print('CSV文件已更新并保存为', output_file_path)