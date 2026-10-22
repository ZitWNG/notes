# L2 pHash 方案 2 修改记录

日期：2026-10-20。

本次实现完整 pHash 直接检索与 8 band 倒排索引共用一份 resized key
映射，并使 pHash、SSIM 和 resized tensor 传输跟随输入设备。
修改在本地工作区完成，未提交、推送，也未改变配置字段或 Memcache key 格式。

## 1. 修改范围

生产代码位于 `vllm_ascend/distributed/ec_transfer/`：

| 文件 | 修改 |
| --- | --- |
| `perceptual_ssim_cache_matcher.py` | 增加 band 索引、零距离直查、复用批量检索、等待并行任务完成、设备输入留在调用线程、补齐最终 SSIM 阈值检查 |
| `phash.py` | 输入设备上 FP32 计算、按设备缓存 DCT 常量、打包 8 字节结果、使用 `int.bit_count()` |
| `ssim.py` | 输入设备上 FP32 计算、按设备缓存高斯核、拒绝不同设备之间隐式比较 |
| `memcache_resized_tensor_store.py` | 去掉强制 CPU 搬移、匹配传输方向与缓冲区设备、保护外部拷贝和缓冲区复用、串行化 PUT 注册事务 |

测试及验证工具：

| 文件 | 内容 |
| --- | --- |
| `tests/ut/distributed/test_perceptual_cache_matcher.py` | 索引、排序、碰撞、失败不发布、批量去重、并发完成、设备线程、SSIM 阈值回归 |
| `tests/ut/distributed/test_phash_tensor.py` | CPU 参考算法对比、数据类型、位序、8 字节结果、设备常量及硬件条件测试 |
| `tests/ut/distributed/test_resized_tensor_devices.py` | 主机真实字节复制、NPU 方向与同步 mock、重复指针 PUT、设备切换、异常清理 |
| `tests/ut/distributed/test_ec_memcache_connector.py` | 将过期 mock 更新为现有 `find_and_cache_matched_ssim()` 的 `str \| None`/异常接口 |
| `benchmarks/benchmark_ec_phash_lookup.py` | 调用生产检索方法，与旧全表排序遍历比较；覆盖均匀分布和共享 band 的退化分布 |

`ec_memcache_connector.py` 本身未修改。

## 2. 方案 2 的数据结构和检索

### 2.1 共享完整 pHash 映射

```text
phash_to_resized_mm_hashes:
    完整 64-bit pHash -> resized key 列表

_phash_band_index:
    (band 位置, 8-bit 段值) -> 完整 pHash 集合
```

resized key 列表只保存在第一份映射中。band 索引存储 pHash 引用，
不会将同一 resized key 列表复制八份。

索引为 matcher 实例内的 CPU Python 字典/集合，生命周期与 matcher 一致。
本次未新增跨进程索引共享、索引持久化或过期清理机制。

### 2.2 `max_hamming == 0`

直接执行 `phash_to_resized_mm_hashes.get(query_phash)`，
返回列表副本，防止调用方修改内部列表。

这一模式不计算 band、不遍历 pHash、不计算汉明距离；
插入时也不填充 band 索引，索引保持为空。
结果仍经过布局兼容性检查、SSIM 和 encoder cache 存在性检查。

### 2.3 `1 <= max_hamming <= 7`

1. 将查询 pHash 按原有位序切成 8 个 8-bit band。
2. 查找 8 个 `(位置, 段值)` 桶，合并去重得到候选 pHash。
3. 使用 `(query ^ candidate).bit_count()` 计算完整汉明距离。
4. 保留距离不大于阈值的 pHash，按 pHash 数值升序排列。
5. 按各 pHash 对应 key 列表的原有顺序展开。

保留原先的候选顺序，是因为 SSIM 会在第一个包含合格候选的批次后提前结束；
随意按集合顺序或汉明距离排序可能改变最终选择。

**召回保证：** 最多 7 个不同 bit 最多影响 7 个 band，因此至少有一个 band
完全相同。当前配置校验范围仍为 0–7；大于等于 8 时该保证不成立。
共享 band 只是粗筛，不能替代完整汉明距离和 SSIM。

配置仍使用 `ec_connector_extra_config.similarity_cache.max_hamming`，默认值为 5。
没有新增 `MAX_HAMMING` 环境变量。配置对象为冻结 dataclass；
调整阈值应重建 matcher，不能在已运行的零距离实例上直接切换成 band 模式。

### 2.4 索引维护与批量查询

- resized tensor 成功写入或确认已存在后，才发布 pHash 和 key。
- 完整映射、band 索引和当前图片的缓存条目在同一把锁内更新。
- 同一 pHash 首次出现时写入 8 个桶；后续相同 pHash 只追加尚不存在的 key。
- Memcache PUT 返回失败或抛出异常时，不发布未成功存储条目的索引。
- 单次查询和批量预热共用 `find_resized_mm_hashes_by_hamming()`。
- 同一预热批次内，相同查询 pHash 只执行一次检索；保留原有结果展开顺序。
- 修复原来未消费 `executor.map()` 结果的问题：提交后等待所有任务，
  再读取结果和传播异常，避免预热返回后仍在写入索引。

### 2.5 复杂度和退化情况

设 `U` 为不同 pHash 数，`P` 为 8 个桶的总 posting 数，`C` 为去重候选数，
`R` 为通过完整汉明距离筛选的 pHash 数，`K` 为输出 key 数。

| 路径 | 查询成本 |
| --- | --- |
| 旧全表排序遍历 | `O(U log U + K)` |
| 零距离直接检索 | 平均 `O(1 + K)` |
| band 检索 | `O(P + C + R log R + K)`，且 `P <= 8C` |

非零距离模式增加约 `8U` 个集合成员，最多 2048 个桶；实际 Python 内存开销
还包括字典和集合容量。零距离模式不承担这些 posting 的维护开销。

均匀、独立的 64-bit hash 假设下，任一 band 命中的概率为
`1 - (255 / 256)^8`，约 3.08%。真实 pHash 具有相关性，不能据此承诺
固定加速比。如果大量 pHash 共享某个 band，`C` 可退化到 `U`。

## 3. pHash 计算与 CPU/NPU 数据流

### 3.1 先澄清现有数据来源

当前请求的 pHash 从 request feature 中现有的 resized `pixel_values`
计算；历史 resized tensor 从 Memcache 读取后用于 SSIM。
本次没有增加“为了计算当前请求 pHash，先从 Memcache 取回当前图片”的往返。

### 3.2 计算跟随输入设备

```text
CPU 输入 -> CPU FP32 灰度 / resize / DCT / 阈值 / 打包 -> Python pHash
NPU 输入 -> NPU FP32 灰度 / resize / DCT / 阈值 / 打包
                                                    -> 8 字节到 CPU
                                                    -> Python 索引查询
```

保留原有 merge 逆置换、按帧均值、32×32 双线性插值、完整 DCT、
左上 8×8 低频、median 与 0.05×std 死区。
统一计算类型为 FP32，不改变原 tensor 的设备、持久化 dtype 或 key。

DCT 矩阵在 CPU 上按原公式生成后，按需要缓存到目标设备；
高斯核也按窗口大小和设备缓存。常量首次放入 NPU 缓存时完成同步，
后续可安全被其他调用流复用。

64 个布尔位在输入设备上打包成 8 个 `uint8`，仅物化一次 8 字节结果，
再按大端位序转为 Python 无符号整数，保留第 63 位。
CPU 索引仍需要主机值，所以 NPU 路径保留每张图片一次结果回传；
本次未实现跨图片批量 pHash 回传。

CPU 输入不会为这个小算子被强制搬到 NPU。
CPU 参考测试确认所测输入与原 FP32 定义一致；
CPU/NPU 的浮点舍入可能不同，本次不承诺跨设备 pHash 逐位一致。

### 3.3 Memcache 传输

| tensor / 缓冲区设备 | PUT | GET |
| --- | --- | --- |
| CPU | `COPY_H2G` | `COPY_G2H` |
| NPU | `COPY_L2G` | `COPY_G2L` |

PUT 使用 `detach().contiguous()`，移除原先强制 `.to("cpu")`。
常驻接收缓冲区和 oversized 临时缓冲区跟随当前请求设备；
切换设备时先释放旧注册，再分配和注册新缓冲区。

NPU 在下列边界进行显式同步：

1. 外部 SDK 读取 PUT 指针之前，完成生产者及 contiguous 操作。
2. SDK 写入 GET 缓冲区之前，完成调用流上的前序使用。
3. 从接收缓冲区 clone 候选后，完成读取再允许下次 GET 覆盖或注销；
   clone 异常路径也会等待已排入流的操作。
4. 设备切换和关闭时，完成旧缓冲区的调用流操作。

这些同步保证外部 SDK 与 PyTorch 的指针生命周期正确，并不意味着零同步。
实现沿用现有 SDK PUT/GET 返回时拷贝已完成的契约；真实传输模式、
主机内存注册和 NPU DMA 能力仍须在部署的 Memcache SDK 上验证。
代码不会自动猜测不支持的设备的拷贝方向。

Memcache backend 具有共享 `_pending_buffers` 状态。
新增实例级 PUT 锁保护“注册、传输、注销”全过程，防止多个 CPU
预处理线程同时注册、或重复 tensor 指针在其他 PUT 尚未完成时被注销。

### 3.4 线程与 SSIM

CPU 输入可使用原有线程池；包含设备 pixel tensor 或 grid tensor 的 pHash
批次留在调用线程。设备上的 SSIM 也留在调用线程，避免线程池丢失调用方
的 NPU 当前流。CPU SSIM 并行任务在失败时也会全部等待完成。

SSIM 的灰度平面和高斯核统一为同设备 FP32。
本次仍保留每对图片的 SSIM 标量读取；它在 NPU 上会同步。
设备 grid 的三个维度转为主机 tuple 也仍需读取。后续可在实机 profile
基础上进一步实现批量 score 回传及计算融合。

## 4. 同时修复 SSIM 最终接受条件

回归测试复现：原 `find_and_cache_matched_ssim()` 在所有批次均未达阈值时，
仍会将非空结果列表的首项作为命中返回。

现在在记录已计算分数后，要求最终分数严格满足
`score > ssim_threshold`。低于阈值、等于阈值、NaN 都不命中。
这也确保 `max_hamming=0` 的 pHash 碰撞不能绕过 SSIM。

## 5. 验证记录

### 5.1 单元测试

在本机 Windows、Python 3.14.3、PyTorch 2.14.1+cpu 的临时环境执行。
使用真实 CPU PyTorch 运算，以隔离 runner 替换缺少的外部 vLLM 导入和
Memcache backend；并绕开依赖完整 Ascend/vLLM 环境的 UT 根 conftest。
这不等同于完整 vLLM、CANN、Memcache 集成测试。

本机执行命令：

```powershell
& 'C:/Users/zongy/AppData/Local/Temp/codex-l2-phash-validation/Scripts/python.exe' `
  .venv/l2-validation/runner.py `
  tests/ut/distributed/test_phash_tensor.py `
  tests/ut/distributed/test_resized_tensor_devices.py `
  tests/ut/distributed/test_perceptual_cache_matcher.py `
  tests/ut/distributed/test_ec_memcache_connector.py -q
```

结果：**99 passed，1 skipped**。跳过项为真实 NPU 非默认流测试。
环境还输出一条缺少 NumPy 的 PyTorch 初始化警告；这些测试不调用 NumPy。
本地 `.venv/l2-validation/runner.py` 是被 Git 忽略的验证辅助文件。

完整项目环境中应直接运行上述四个测试文件的 `pytest`，
并在真实 Ascend 环境运行设备测试。新增测试覆盖：

- 所有阈值 0–7，与全量暴力检索的有序结果比较。
- 零阈值禁止扫描、禁止 band/Hamming 运算的结构性性能回归。
- band 候选去重、假阳性过滤、64 位最高位、碰撞、重复注册、空结果。
- PUT 失败不发布索引、批量相同查询只检索一次、工作线程异常传播。
- 原 CPU FP32 算法、FP16/FP32/FP64 输入、常量图片、8 字节物化。
- CPU 实际字节回填和 clone 后的缓冲区覆盖隔离。
- NPU 传输方向与同步顺序 mock、设备切换、clone 失败清理。
- pHash 相同但 SSIM 不合格的拒绝。

开发过程先运行新增用例复现旧行为，再修改实现；独立复查同时检查了
索引召回/顺序、PUT 生命周期和 NPU 流边界。

### 5.2 CPU 索引微基准

在具备项目运行依赖的环境中：

```bash
PYTHONPATH=. python benchmarks/benchmark_ec_phash_lookup.py \
  --sizes 1000 10000 --queries 64 --repeats 3
```

本机通过相同隔离 runner 的 `--script` 模式执行。
使用固定随机种子，查询一半选已登记 hash、一半新生成；
每组先核对有序结果，再取三轮每查询平均时间的中位数。
以下为 10000 个不同 pHash 的示例结果：

| 分布 | 阈值 | 平均候选 pHash 数 | 原全表检索 μs/次 | 新检索 μs/次 |
| --- | --- | --- | --- | --- |
| 均匀随机 | 0 | 0.5 | 5035.78 | 0.29 |
| 均匀随机 | 5 | 307.7 | 4817.61 | 46.18 |
| 所有 hash 共享一个 band | 0 | 0.5 | 4441.84 | 0.23 |
| 所有 hash 共享一个 band | 5 | 10000.0 | 4778.82 | 1055.78 |

这只是本机合成数据上的 Python 索引微基准，排除了图片计算、SSIM、
Memcache、NPU 和线程池，不代表真实图片分布、端到端吞吐或生产延迟。
共享 band 测试明确展示了候选集可能退化为全量。

### 5.3 静态检查与实机待验项

对修改文件运行 Ruff 检查/格式检查、PyMarkdown 和 `git diff --check`。
完整 `bash format.sh ci` 未运行：当前环境缺少 Bash/pre-commit 及全套 hook
工具；定向检查不能替代该项目级检查。

本机没有 Ascend NPU 和真实 Memcache 服务，因此尚未验证：

1. 项目目标 PyTorch/torch_npu/CANN 版本下的算子支持和实际执行设备，
   包括 median、std、整数打包、非默认流及多设备。
2. 所用 Memcache transport 的主机/NPU 注册和传输能力，以及同步返回契约。
3. CPU/NPU 结果一致性、真实图片 SSIM 阈值与召回率。
4. 端到端时延、NPU 同步占比、峰值 HBM/主机内存和真实索引桶分布。

建议实机记录 pHash 计算、索引召回、resized GET、SSIM 四阶段耗时，
并同时记录候选数量和命中率，再评估是否值得继续批量化 NPU 计算。
