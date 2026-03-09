# %%
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
from skimage.restoration import denoise_tv_chambolle
from scipy.optimize import curve_fit
import numpy as np

def check_and_clean_flicker_1d(timeseries, max_flips=3, min_active_years=2):
    """
    针对单像素的一维时间序列进行闪烁噪声检查。
    如果判定为噪声，返回全0数组；否则返回原数组。
    """
    # 1. 二值化
    # 只要大于0.1就视为"有建筑物" (防止极小浮点数误差)
    binary = timeseries > 0.1
    
    # 2. 计算翻转次数 (Flip Count)
    # [1, 0, 0] -> diff [-1, 0] -> abs [1, 0] -> sum 1
    flips = np.sum(np.abs(np.diff(binary.astype(int))))
    
    # 3. 计算活跃年份 (Active Years)
    active_years = np.sum(binary)
    
    # --- 判定逻辑 ---
    
    # 条件A: 翻转太频繁 (e.g. 0->30->0->30->0, flips=4)
    # 这通常是混合像元或边缘噪声
    is_unstable = flips > max_flips
    
    # 条件B: 出现次数太少 (孤立点)
    # 默认判定：如果总共出现的年份少于阈值，则视为稀有噪声
    is_rare = active_years < min_active_years
    
    # --- 【关键修改】端点保护逻辑 ---
    # 如果 active_years < 2，通常认为是噪声（比如中间某一年突然出现又消失）。
    # 但是，如果这一年出现在"序列开头"或"序列结尾"，且没有发生多次翻转，
    # 则可能是"2015年后拆迁"或"2024年新建"，这是合法的。
    
    if is_rare and (active_years > 0):
        # 情况1: 2024年新建 (End-point Construction)
        # 逻辑: 最后一年有值，且翻转次数很低(通常为1: 0->1)
        is_valid_new_build = (binary[-1] > 0) and (flips <= 1)
        
        # 情况2: 2015年后立即拆迁 (Start-point Demolition)
        # 逻辑: 第一年有值，且翻转次数很低(通常为1: 1->0)
        is_valid_demolition = (binary[0] > 0) and (flips <= 1)
        
        # 如果满足任一端点保护条件，则豁免
        if is_valid_new_build or is_valid_demolition:
            is_rare = False 

    # --- 执行清洗 ---
    # 必须同时满足"不稳定"或"稀有且未被豁免"，才会被置0
    if is_unstable or is_rare:
        # 调试用：
        # print(f"检测到噪声 -> 置0 (Flips={flips}, Active={active_years})")
        return np.zeros_like(timeseries)
    else:
        return timeseries

# ==========================================
# 1. 算法一：中值滤波 + 逻辑修正 (推荐方案)
# ==========================================
def algo_pure_monotonic(data, demo_thresh=15.0):
    """
    [最终简化版] Logic V4: 纯粹单调性约束 + 拆迁保护
    
    策略：
    1. 既然取消了尖峰剔除，意味着我们完全信任模型的"高值"预测。
    2. 只有当数值"剧烈下降且接近0"时，才认为是拆迁，允许重置高度。
    3. 其他任何波动，一律强制拉平到前一年的高度（只增不减）。
    """
    # 0. 基础清洗 (Flicker Clean) - 这一步必须保留，处理Mask的分类错误
    # 请确保 check_and_clean_flicker_1d 已经在上下文中定义
    if np.nansum(data) == 0: return np.zeros_like(data)
    data = check_and_clean_flicker_1d(data, max_flips=3, min_active_years=2)
    
    n = len(data)
    final = data.copy()
    
    # 单向扫描
    for t in range(1, n):
        prev = final[t-1]
        curr = final[t]
        
        # 情况A: 高度增加 (或者不变) -> 直接信任，更新高度
        if curr >= prev:
            pass 
        
        # 情况B: 高度下降
        else:
            diff = prev - curr
            
            # 判断是否为拆迁: 下降幅度大 AND 当前值很低
            # 这里的 < 5.0 是为了容忍地表的一点点残余高度误差
            is_demolition = (diff > demo_thresh) and (curr < 5.0)
            
            if is_demolition:
                # 是拆迁，允许下降 (重置基准)
                # final[t] = curr  <-- 也就是保留这个低值
                pass
            else:
                # 不是拆迁，仅仅是预测波动 -> 强制拉平
                # 无论后面怎么跌，都锁死在 prev 的高度
                final[t] = prev
                
    return final

# ==========================================
# 2. 算法二：PELT + 残差召回 (混合策略)
# ==========================================
def algo_epelt(data, penalty=2.0, demo_thresh=5.0):
    if np.nansum(data)==0:
        return np.zeros_like(data)
    data = check_and_clean_flicker_1d(data)
    # PELT 变点检测
    algo = rpt.Pelt(model="l2", min_size=1).fit(data)
    breakpoints = algo.predict(pen=penalty)
    
    # 构建拟合线
    pelt_fit = np.zeros_like(data)
    start_idx = 0
    for end_idx in breakpoints:
        # 使用段内中值作为该段的高度
        val = np.median(data[start_idx:end_idx])
        pelt_fit[start_idx:end_idx] = val
        start_idx = end_idx
        
    # 残差召回 (Rescue)
    final = pelt_fit.copy()
    residuals = final - data
    # 逻辑：原始数据很小 AND 拟合数据很大 -> 说明PELT填平了坑 -> 强制恢复
    is_missed = (data < demo_thresh) & (residuals > 10.0)
    final[is_missed] = data[is_missed]
    return final

# ==========================================
# 3. 算法三：TVD (全变分去噪)
# ==========================================
def algo_tvd(data, weight=5):
    if np.nansum(data)==0:
        return np.zeros_like(data.shape)
    data = check_and_clean_flicker_1d(data)

    """
    Total Variation Denoising.
    weight: 调节平滑度。越大越平，越小越接近原始数据。
    """
    # scikit-image 的 denoise_tv_chambolle 实现了 TVD
    # 这是一个数学优化过程，天然形成阶梯状
    return denoise_tv_chambolle(data, weight=weight)

# ========================================== 
# 4. 算法四：LandTrendr
# ==========================================
def algo_landtrendr_proxy(data, epsilon=5.0):
    if np.nansum(data) == 0: 
        return np.zeros_like(data)
    data = check_and_clean_flicker_1d(data) # 同样先清洗
    
    n = len(data)
    x = np.arange(n)
    
    best_err = np.inf
    best_fit = np.zeros_like(data)
    
    # 暴力搜索最佳转折年份 k
    for k in range(1, n-1):
        # 定义折线函数：在 k 处转折
        def piecewise_linear(t, y0, y_k, y_end):
            cond = t <= k
            # 第一段: 0 -> k
            seg1 = y0 + (y_k - y0) * (t / k) 
            # 第二段: k -> end
            seg2 = y_k + (y_end - y_k) * ((t - k) / (n - 1 - k))
            return np.where(cond, seg1, seg2)
        
        try:
            # 拟合找到最佳的 y0, y_k, y_end
            popt, _ = curve_fit(piecewise_linear, x, data, p0=[data[0], data[k], data[-1]], maxfev=1000)
            y_fit = piecewise_linear(x, *popt)
            err = np.sum((data - y_fit)**2)
            if err < best_err:
                best_err = err
                best_fit = y_fit
        except:
            continue
            
    # 同时也试一下不转折（纯直线），看谁误差小
    z = np.polyfit(x, data, 1)
    y_lin = np.poly1d(z)(x)
    if np.sum((data - y_lin)**2) < best_err:
        return y_lin
        
    return best_fit

# ==========================================
# 5. 算法五：HMM (隐马尔可夫模型 - Viterbi解码)
# ==========================================
def algo_hmm(data):
    if np.nansum(data)==0:
        return np.zeros_like(data.shape)
    data = check_and_clean_flicker_1d(data)
    """
    自定义简单的 HMM。
    状态空间: 0m 到 100m (整数)
    观测: 原始数据
    """
    states = np.arange(0, 101) # 0, 1, ..., 100
    n_states = len(states)
    n_obs = len(data)
    
    # 1. 初始概率 (假设均匀分布)
    prior = np.ones(n_states) / n_states
    
    # 2. 转移矩阵 (Transition Probability)
    # 定义物理规则：倾向于保持，倾向于变高，极难变矮(除非归零)
    trans_mat = np.zeros((n_states, n_states))
    for i in range(n_states): # from state i
        for j in range(n_states): # to state j
            if i == j: 
                prob = 0.8  # 保持原样的概率最高
            elif j > i:
                prob = 0.15 # 变高的概率
            elif j == 0 and i > 10:
                prob = 0.05 # 突然拆迁归零的概率
            elif j < i:
                prob = 0.0001 # 变矮(非归零)的概率极低 (惩罚)
            else:
                prob = 0.0
            trans_mat[i, j] = prob
            
    # 归一化转移矩阵
    trans_mat = trans_mat / trans_mat.sum(axis=1, keepdims=True)
    
    # 3. 发射概率 (Emission Probability)
    # 假设观测值服从高斯分布 P(Obs|State) ~ N(State, sigma)
    sigma = 10.0 # 模型的观测噪声容忍度
    emission_mat = np.zeros((n_states, n_obs))
    for t in range(n_obs):
        # 计算所有状态生成当前观测值 data[t] 的概率 PDF
        emission_mat[:, t] = np.exp(-0.5 * ((data[t] - states) / sigma)**2)

    # 4. Viterbi 算法 (动态规划求解最优路径)
    # dp[t, s] 表示时刻t处于状态s的最大概率
    dp = np.zeros((n_obs, n_states))
    path = np.zeros((n_obs, n_states), dtype=int)
    
    # 初始化
    dp[0, :] = prior * emission_mat[:, 0]
    
    # 递归
    for t in range(1, n_obs):
        for s in range(n_states):
            # 前一时刻所有状态到当前状态s的概率 * 发射概率
            prob_trans = dp[t-1, :] * trans_mat[:, s]
            best_prev = np.argmax(prob_trans)
            path[t, s] = best_prev
            dp[t, s] = prob_trans[best_prev] * emission_mat[s, t]
            
        # 防止下溢 (Normalize)
        dp[t, :] /= dp[t, :].sum()

    # 回溯
    best_path = np.zeros(n_obs, dtype=int)
    best_path[-1] = np.argmax(dp[-1, :])
    for t in range(n_obs-2, -1, -1):
        best_path[t] = path[t+1, best_path[t+1]]
        
    return states[best_path]

# ==========================================
# 算法比较
# ==========================================
def comparison(raw_data, years=np.arange(2015, 2025), fig_savepath='', legend=False):
    # 计算所有结果
    res_median = algo_pure_monotonic(raw_data)
    res_pelt = algo_epelt(raw_data, penalty=2) # 调参 penalty
    res_tvd = algo_tvd(raw_data, weight=10) # 调参 weight: 越大越阶梯
    res_landtrendr = algo_landtrendr_proxy(raw_data)
    res_hmm = algo_hmm(raw_data)

    # 绘图配置
    plt.figure(figsize=(2.5, 8)) 
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 13

    # 子图 1: 原始数据
    ax1 = plt.subplot(6, 1, 1)
    ax1.plot(years, raw_data, 'k--', linewidth=2.5, alpha=0.5, label='Original prediction')
    if legend:
        ax1.legend(loc='upper left', frameon=False, prop={'weight':'bold'})
    ax1.set_ylim(0, 100)
    ax1.set_xlim(2014, 2025)
    ax1.set_xticks([2015, 2024])
    ax1.set_xticklabels([])
    # ax1.spines['top'].set_visible(False)  
    # ax1.spines['right'].set_visible(False)  

    # 子图 2: 中值滤波 + 逻辑
    ax2 = plt.subplot(6, 1, 2, sharex=ax1, sharey=ax1)
    ax2.plot(years, raw_data, 'k--', alpha=0.4)
    ax2.plot(years, res_median, color='#D55E00', linewidth=2, marker='o', markersize=5, label='Median filtering')
    if legend:
        ax2.legend(loc='upper left', frameon=False, prop={'weight':'bold'})
    ax2.set_ylim(0, 100)
    ax2.set_xticks([2015, 2024])
    ax2.set_xticklabels([])
    # ax2.spines['top'].set_visible(False)  
    # ax2.spines['right'].set_visible(False)  

    # 子图 3: PELT + Rescue
    ax3 = plt.subplot(6, 1, 3, sharex=ax1, sharey=ax1)
    ax3.plot(years, raw_data, 'k--', alpha=0.4)
    ax3.plot(years, res_pelt, color='#0072B2', linewidth=2, marker='o', markersize=5, label='EPELT')
    if legend:
        ax3.legend(loc='upper left', frameon=False, prop={'weight':'bold'})
    ax3.set_xticklabels([])
    ax3.set_ylim(0, 100)
    ax3.set_xticks([2015, 2024])
    ax3.set_xticklabels([])
    # ax3.spines['top'].set_visible(False)  
    # ax3.spines['right'].set_visible(False)  

    # 子图 4: TVD
    ax4 = plt.subplot(6, 1, 4, sharex=ax1, sharey=ax1)
    ax4.plot(years, raw_data, 'k--', alpha=0.4)
    ax4.plot(years, res_tvd, color='#009E73', linewidth=2, marker='o', markersize=5, label='TVD')
    ax4.set_xticklabels([])
    if legend:
        ax4.legend(loc='upper left', frameon=False, prop={'weight':'bold'})
    ax4.set_ylim(0, 100)
    ax4.set_xticks([2015, 2024])
    ax4.set_xticklabels([])
    # ax4.spines['top'].set_visible(False)  
    # ax4.spines['right'].set_visible(False)  

    # 子图 5: LandTrendr (Proxy)
    ax5 = plt.subplot(6, 1, 5, sharex=ax1, sharey=ax1)
    ax5.plot(years, raw_data, 'k--', alpha=0.4)
    ax5.plot(years, res_landtrendr, c='#CC79A7', linewidth=2, marker='o', markersize=5, label='LandTrendr')
    ax5.set_xticklabels([])
    ax5.set_ylim(0, 100)
    if legend:
        ax5.legend(loc='upper left', frameon=False, prop={'weight':'bold'})
    ax5.set_xticks([2015, 2024])
    ax5.set_xticklabels([])
    # ax5.spines['top'].set_visible(False)  
    # ax5.spines['right'].set_visible(False)  

    # 子图 6: HMM
    ax6 = plt.subplot(6, 1, 6, sharex=ax1, sharey=ax1)
    ax6.plot(years, raw_data, 'k--', alpha=0.4)
    ax6.plot(years, res_hmm, c='#E69F00', linewidth=2, marker='o', label='HMM')
    if legend:
        ax6.legend(loc='upper left', frameon=False, prop={'weight':'bold'})
    ax6.set_ylim(0, 100)
    ax6.set_xticks([2015, 2024])
    ax6.set_xticklabels([])
    # ax6.spines['top'].set_visible(False)  
    # ax6.spines['right'].set_visible(False)  
    plt.savefig(fig_savepath, dpi=300, transparent=True, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    comparison(np.array([15, 22, 18, 20, 0, 82, 90, 81, 100, 80], dtype=float), fig_savepath='..\\case_study\\sensitivity of post-process\\example-1.png')
    comparison(np.array([0, 3, 4, 0, 56, 76, 69, 83, 85, 76], dtype=float), fig_savepath='..\\case_study\\sensitivity of post-process\\example-2.png')
    comparison(np.array([80, 82, 79, 81, 70, 80, 83, 86, 81, 79], dtype=float), fig_savepath='..\\case_study\\sensitivity of post-process\\example-3.png')
    comparison(np.array([16, 23, 18, 21, 21, 18, 3, 2, 0, 1], dtype=float), fig_savepath='..\\case_study\\sensitivity of post-process\\example-4.png')
    comparison(np.array([0, 30, 0, 25, 5, 0, 40, 0, 10, 0], dtype=float), fig_savepath='..\\case_study\\sensitivity of post-process\\example-5.png')


