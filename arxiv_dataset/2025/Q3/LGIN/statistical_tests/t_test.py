import scipy.stats as stats

group_lgin = [83.2, 82.3]
group_gin = [86, 86]

t_statistic, p_value = stats.ttest_rel(group_lgin, group_gin)

alpha = 0.1  # 10% significance

if p_value <= alpha:
    print("Reject null hypothesis")
    print("There is a significant difference between the two groups.")
else:
    print("Fail to reject null hypothesis")
    print("There is no significant difference between the two groups.")

print("T-statistic:", t_statistic)
print("P-value:", p_value)
