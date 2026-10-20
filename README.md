# locbench — 定位算法测试 / 调参（纯数值模拟）

对声发射震源定位算法（aelib 的 `GeigerLocator` 及任何注册到
`LOCATORS` 的定位器）做**合成数据**测试与参数调优的实验层。
**本库不含任何实测数据**：台阵、震源、到时全部由数值模拟生成
（到时误差、采样率量化、丢道均可注入），真值已知，误差可直接统计。

依赖方向：`locbench → aelib`（永不反向）。算法本体在
[aelib](../aelib) 中维护；本库只做评估与可视化。

## 安装（前提：环境已装 aelib）

aelib 为私有库、不经 pip 分发——**安装本库的环境必须已装有 aelib**
（`aelib>=0.9.39`；缺失时安装会以"找不到 aelib"停止，即提示先装 aelib）：

```bash
pip install git+https://github.com/glassyskyabove/locbench.git
# 本地开发改为： pip install -e E:\GitHub_codes\locbench
```

依赖（numpy/pandas/matplotlib/streamlit）自动安装；装完即有 `locbench`
命令。

## 可视化界面

```bash
locbench                                  # 或 streamlit run locbench/app.py
```

侧栏配置合成场景：维度（三维/二维）× 传感器布置——**几何特征**（横边/
纵边/垂边=与 x/y/z 平行的全部边各布等分点，侧面/正面/顶面=以 x/y/z 为
法向的两面各取面心，顶点=全部角点，可任意组合）、随机整数坐标、或自定
义文本（每行一个传感器、坐标逗号分隔、单位随显示单位）；默认三维
100×100×200 mm 试样、几何特征勾选顶点（8 角点）。尺寸/间距单位 mm|m、
到时误差单位 µs|ms、采样率单位 MHz|kHz（默认 mm/µs/MHz）；震源可随机
或按 xyz 间距网格布点（覆盖试样盒，与传感器包围盒解耦）；到时误差 σ、
采样率、丢道、真实波速与定位器参数（采用波速/初值点个数/外延比例/迭代
上限/收敛判据 tol（默认 1/fs，随采样率联动）/LM 阻尼/联合反演速度），
两个页签：

- **场景与结果**：定位误差中位/均值/p95、收敛率、残差 rms、迭代数、
  耗时；台阵+真值+定位结果的等轴 3D/2D 图（误差向量可视化）、误差
  分布直方图；扫描取值支持等间距简写（如 `2600:8000:300`）；
- **参数敏感性**：对定位器参数（grid_size/margin/velocity/max_iter/
  tol）或场景参数（σ/采样率/丢道）做单参数扫描，画误差曲线（中位线 +
  四分位带）与耗时曲线；tol 用对数轴；结果按哈希落盘缓存（7 天自动
  清理）。

## 脚本用法

```python
from aelib import GeigerLocator
from locbench import (Scenario, run_trials, summarize, sweep,
                      random_layout, random_sources)

sensors = random_layout(8, extent=(1, 1, 1), seed=0)     # 米
sources = random_sources(20, sensors, seed=1)
sc = Scenario(sensors, sources, velocity=5300.0,
              sigma=1e-6, repeats=5, seed=2)             # 1 µs 拾取噪声

df = run_trials(GeigerLocator(sensors, velocity=5300.0), sc)
print(summarize(df))          # 误差中位/p95、收敛率、rms、迭代、耗时…

# 参数网格扫描（结果按 场景+网格+定位器+aelib版本 哈希落盘缓存）
df2 = sweep(GeigerLocator, {"grid_size": [8, 16, 24, 48],
                            "margin": [0.02, 0.05, 0.1]},
            sc, locator_kwargs={"velocity": 5300.0})
```

## 结构

```
locbench/
├── synth.py       合成数据：random_layout / coplanar_layout /
│                  random_sources / make_arrivals（噪声+采样率量化+丢道）
├── evaluate.py    Scenario / run_trials（逐试验指标表）/ summarize
├── sweep.py       参数网格扫描 + 磁盘缓存（results/）
├── plots.py       matplotlib 图：布局+误差向量 / 指标-参数曲线 / 热力图
└── app.py         streamlit 交互调参界面（locbench 命令启动）
```

## 约定

- 单位一律 SI（米、秒、m/s）：合成数据本身就是米制普通字典，
  不涉及 aelib 的声明单位契约（那是实测数据入口的事）。
- 指标口径：`err` 为真误差（欧氏距离）；`stderr_calibration`
  = 真误差 / 形式误差的中位数，≈1 表示协方差可信，>1 偏乐观。
- 新定位算法无需改本库：在运行时 `@LOCATORS.register("my_locator")`
  注册后把 `sweep` 的 `locator_cls` 换成你的类即可。

## 测试

```bash
pytest tests/ -q
```
