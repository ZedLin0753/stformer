import matplotlib.pyplot as plt

# 設置流程圖節點
fig, ax = plt.subplots(figsize=(6, 12))
ax.set_xlim(0, 1)
ax.set_ylim(0, 9)
ax.axis("off")

# 定義節點位置與標籤
nodes = [
    (0.5, 8, "X_in"),  
    (0.5, 7, "Projection to QKV"),  
    (0.5, 6, "Spatial Multi-Head Attention"),  
    (0.5, 5, "Projection out Attention (Spatial)"),  
    (0.5, 4.5, "Residual Connection (Spatial)"),  
    (0.5, 4, "Feed Forward (Spatial)"),  
    (0.5, 3, "Projection to QKV (Temporal)"),  
    (0.5, 2, "Temporal Multi-Head Attention"),  
    (0.5, 1, "Projection out Attention (Temporal)"),  
    (0.5, 0.5, "Residual Connection (Temporal)"),  
    (0.5, 0, "Feed Forward (Temporal)"),  
    (0.5, -1, "X_out")  
]

# 繪製節點
for x, y, label in nodes:
    ax.text(x, y, label, ha="center", va="center", fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightblue"))

# 繪製箭頭
for i in range(len(nodes) - 1):
    x1, y1, _ = nodes[i]
    x2, y2, _ = nodes[i + 1]
    ax.annotate("", xy=(x2, y2 + 0.1), xytext=(x1, y1 - 0.1),
                arrowprops=dict(arrowstyle="->", color="black"))

# 顯示流程圖
plt.title("STA_Block Transformer Flowchart (Without Inter-Frame Aggregation)")
plt.show()
