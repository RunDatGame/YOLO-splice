import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 设置全局字体为 Times New Roman，小四号（12pt）
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16  # 小四号字体约等于 12 pt


# 模拟归一化混淆矩阵数据（与你上传的图一致）
conf_matrix = np.array([
    [0.92, 0.00, 0.53],
    [0.00, 0.95, 0.47],
    [0.08, 0.05, 0.00]
])

# 类别标签
labels = ['FS', 'PL', 'background']

# 绘图
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels, fmt='.2f')

plt.title("Confusion Matrix")
plt.xlabel("True")
plt.ylabel("Predicted")
plt.show()
