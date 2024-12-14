import matplotlib.pyplot as plt
import numpy as np

# 读取数据
data = np.loadtxt('histogram_data.txt')

# 创建直方图
plt.figure(figsize=(12, 6))
plt.bar(data[:,0], data[:,1], width=1.0)
plt.title('CUDA Histogram Distribution (N=1024, Block=256)')
plt.xlabel('Bin Index')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

# 添加一些统计信息
plt.text(0.02, 0.95, f'Total Bins: {len(data)}', 
         transform=plt.gca().transAxes)
plt.text(0.02, 0.90, f'Mean Count: {data[:,1].mean():.2f}', 
         transform=plt.gca().transAxes)
plt.text(0.02, 0.85, f'Max Count: {data[:,1].max()}', 
         transform=plt.gca().transAxes)

# 保存图片
plt.savefig('histogram.png')
plt.close()
