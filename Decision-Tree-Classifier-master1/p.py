import scipy.stats as stats

# 定义两组数据
group1 = [3, 4, 5, 5, 3, 5, 4, 5, 5, 5]
group2 = [3, 3, 2, 3, 3, 5, 2, 4, 4, 4]

# 独立样本 t 检验
t_stat, p_value = stats.ttest_ind(group1, group2)

# 输出结果
print(f"t 统计量: {t_stat:.4f}")
print(f"P 值: {p_value:.4f}")