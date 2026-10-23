# L2 pHash 优化：四个生产文件的差异与修改逻辑

整理日期：2026-10-23。

本文对照固定提交解释代码，记录讨论中确认的行为和仍待实现的要求。
不将算法推导或模拟测试结果表述为真实 NPU 性能结论。

## 1. 对照版本与当前状态

| 项目 | 版本或位置 |
| --- | --- |
| 上游基线 | `hanxi-java/vllm-ascend:l2`，提交 `5fecef105a061c5692e2f727bf5eeed3297a224b` |
| 已提交的优化版本 | `ZitWNG/vllm-ascend:codex/phash-on-l2`，提交 `44d109a9a3509da0c5ec89de8ec92d43f9492770` |
| 本地 worktree | `C:\Users\zongy\Downloads\vllm-ascend-phash-sync` |
| 生产代码目录 | `vllm_ascend/distributed/ec_transfer/` |

本次已提交的优化版本直接基于上述上游提交。比较对象不是旧 `emb-L2`
分支，也不是未来可能继续变化的上游分支头。

写作文档时，worktree 中存在已有的未提交修改。本文以固定提交的语义差异为主，
并重新核对了本地 `ssim_score()` 的设备判断和 matcher 的阈值判断。

| 讨论事项 | 已提交版本及写作时本地行为 | 讨论结论 |
| --- | --- | --- |
| SSIM 异设备输入 | `a.device != b.device` 时返回 `None` | 用户已要求允许异设备输入；拟将 `b` 转到 `a.device`，但这项代码变更尚未写入目标 worktree |
| SSIM 阈值 | 各相关判断统一为严格 `>` | 上游混用 `>` 与 `>=`；本次选择严格 `>`，没有另外决定改成统一 `>=` |

本文只记录差异与设计，不代表上述待实现事项已经完成。

## 2. 四个文件如何配合

| 文件 | 主要职责 | 相对上游的实质变化 |
| --- | --- | --- |
| `phash.py` | 生成 64 位 pHash，提供汉明距离及分段工具 | 按输入设备执行 FP32 计算、按设备缓存常量、设备端打包 8 字节结果、使用 `bit_count()` |
| `ssim.py` | 重建 patch 灰度平面，计算结构相似度 | 按设备缓存高斯核、明确 FP32、增加设备一致性检查；异设备自动对齐属于待实现要求 |
| `perceptual_ssim_cache_matcher.py` | 管理索引、召回候选、协调评分与结果缓存 | 零距离直查、八段索引、批内复用、扩展状态锁保护范围、调整线程池使用条件、统一阈值 |
| `memcache_resized_tensor_store.py` | 存取 resized tensor，组织批次并评分 | 传输与缓冲区跟随设备、保护 PUT 注册生命周期、补充 NPU 同步、修正空分数及返回排序 |

需要区分三种数据：

- 原始 `mm_hash`：用于原有的精确 encoder 缓存查找。
- `pHash`：从图像预处理张量计算，用于召回可能相似的图片。
- resized tensor：候选图像的预处理张量，读取后用于 SSIM；它不是 encoder embedding。

需要重新进行相似匹配时，主要路径为：

```text
当前图像的预处理张量
    → 计算或取得 pHash
    → 完整 pHash 字典查询 / 八段索引召回
    → 完整汉明距离过滤
    → model、grid、shape、dtype 兼容性检查
    → 从 Memcache 分批读取候选 resized tensor
    → 计算 SSIM、排序、按阈值选择候选
    → connector 检查候选 embedding 是否可用并安排复用
```

原始 `mm_hash` 精确命中、已有合格 SSIM 缓存命中均有更短的处理路径。
`pHash` 相同不等于图片逐像素相同，也不意味着可以跳过 SSIM。

## 3. phash.py：设备、精度与 64 位结果打包

源码：[上游 phash.py][up-phash]、[优化版本 phash.py][new-phash]。

### 3.1 原有算法保持了什么

上游已经使用以下过程：

1. 根据 `(t, gh, gw)` 和 `merge_size` 反解 Qwen2/3-VL 的 patch 排列。
2. 对 patch 内部求均值，得到 `(t, gh, gw)` 平面。
3. 对视频时间维求均值，再双线性插值到 `32 × 32`。
4. 计算二维 DCT，取左上 `8 × 8` 低频系数。
5. 与中位数比较，并应用 `_DEAD_ZONE = 0.05` 的死区，生成 64 个 bit。

DCT、低频截取、中位数死区、视频时间维平均都不是本次新增。
这里的灰度平面来自 patch 均值，并不是原始 RGB 图片的标准灰度变换；
结合死区规则，不能宣称它与 `imagehash.phash` 完全一致。

### 3.2 明确在输入设备上使用 FP32

当前代码在重排和求均值前执行：

```python
x = (
    resized.detach()
    .to(dtype=torch.float32)
    .reshape(t, gh // merge_size, gw // merge_size, merge_size, merge_size, -1)
)
```

- `detach()` 避免为缓存相似性计算构建梯度关系。
- 显式 FP32 使求均值、插值和 DCT 的输入精度明确。
- 未指定新的 device，因此 CPU 输入保留在 CPU，NPU 输入保留在 NPU。

上游虽然注释声称“全部 CPU float32”，但函数内部并未主动把输入转换成
CPU FP32。不能把旧注释当成实际设备迁移行为。

FP32 转换时机与 CPU/NPU 的浮点舍入可能影响结果，不能保证所有输入在不同
设备上生成逐位相同的 pHash。

### 3.3 为什么 arange 显式使用 device="cpu"

DCT 常量构造改为：

```python
k = torch.arange(n, dtype=torch.float32, device="cpu").unsqueeze(1)
i = torch.arange(n, dtype=torch.float32, device="cpu").unsqueeze(0)
m = torch.cos(math.pi / n * (i + 0.5) * k)
m[0] *= 1.0 / math.sqrt(2.0)
m = m.to(device=device)
```

当 `n = 32` 时，`k.shape == (32, 1)`，`i.shape == (1, 32)`，
广播后得到 `m.shape == (32, 32)`，FP32 数据约 4 KiB。

显式在 CPU 构造有三个作用：

1. 不受 `torch.set_default_device()` 或默认设备上下文影响。
2. 将坐标、余弦及缩放等小规模常量计算集中在 CPU，之后整体复制到目标设备，
   避免分别在 NPU 启动这些小操作；实际耗时仍应测量。
3. 各设备使用 CPU 构造后复制的常量，减少常量生成方式引入的设备差异。

实际 `m @ img @ m.T` 仍在输入设备上执行。CPU 构造常量不等于把图像计算搬到 CPU。
常量一致也不能保证后续设备算子的浮点结果完全一致。

`_dct_matrix()` 从单份缓存改为按设备缓存，最多 16 条。
NPU 常量首次准备完成后同步当前 stream，再返回供后续调用复用；缓存命中时不重新构造。

### 3.4 为什么改变 64 位结果的打包方式

上游先将 64 个布尔值转成 Python 列表，再逐位组合：

```python
phash = 0
for b in bits.tolist():
    phash = (phash << 1) | int(b)
```

当前先在张量所在设备上压成 8 个字节：

```python
packed = (
    (bits.reshape(-1, _BITS_PER_BYTE).to(torch.int32) * _byte_weights(bits.device))
    .sum(dim=1, dtype=torch.int32)
    .to(torch.uint8)
)
return int.from_bytes(bytes(packed.cpu().tolist()), byteorder="big")
```

| 对比项 | 上游 | 当前 |
| --- | --- | --- |
| 返回 CPU 前的表示 | 64 个 `torch.bool` 元素 | 8 个 `torch.uint8` 元素 |
| Python 列表内容 | 64 个布尔值 | 8 个字节数值 |
| 构造 Python 整数 | 64 次 Python 移位循环 | `int.from_bytes()` |
| 是否仍需取得设备结果 | 是 | 是 |

64 个 bool 元素通常占 64 字节，压缩后数据载荷为 8 字节。
但旧实现不是“64 次 NPU→CPU 同步”：`.tolist()` 取得数据后，逐位循环在 Python 中进行。
新版也仍有 `.cpu().tolist()` 的结果等待。

新版增加了设备上的类型转换、乘法和归约；对于这么小的数据，算子启动开销可能
抵消部分收益。不能仅依据 64 字节变 8 字节就宣称延迟降低 8 倍。

### 3.5 _byte_weights 和 packed 分别是什么

`_byte_weights(device)` 返回形状 `(8,)` 的 int32 张量：

```python
tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.int32, device=device)
```

它由 `1 << bit`、`bit = 7, 6, ..., 0` 构造，表示一个字节从高位到低位的权重。
权重按设备缓存，设备与 `bits` 一致。

打包过程为：

```text
(64,) bool
    → reshape 成 (8, 8)
    → 转成 int32，按行广播乘以 [128, 64, 32, 16, 8, 4, 2, 1]
    → 每行求和，得到 (8,) int32，数值范围 0～255
    → 转成 (8,) uint8，即 packed
```

例如第一组 bit 为 `10110010`：

```text
1×128 + 0×64 + 1×32 + 1×16 + 0×8 + 0×4 + 1×2 + 0×1
= 178 = 0xB2
```

若全部 bit 为：

```text
10110010 00000000 00000000 00000000 00000000 00000000 00000000 00000001
```

则 `packed` 为：

```python
tensor([178, 0, 0, 0, 0, 0, 0, 1], dtype=torch.uint8, device=bits.device)
```

`byteorder="big"` 将首字节放在整数最高位，得到 `0xB200000000000001`，
保留旧实现的 bit 顺序。无需先把整个结果放入有符号 `torch.int64`，
Python 整数可以保留最高位为 1 的无符号 64 位值。

### 3.6 汉明距离与分段工具

`hamming()` 从 `bin(a ^ b).count("1")` 改为 `(a ^ b).bit_count()`，
语义仍是统计异或结果中的 1，减少字符串构造和遍历。

`bands()` 原本就在上游；新增的是 matcher 将它接入实际索引与查询流程。

## 4. ssim.py：同设备计算与异设备需求

源码：[上游 ssim.py][up-ssim]、[优化版本 ssim.py][new-ssim]。

### 4.1 保持不变的评分逻辑

`build_patch_gray_plane()` 按相同的 merge 逆置换重建 patch 平面，
对 patch 内部及时间维求均值，返回 `(1, 1, gh, gw)`。
SSIM 不再把这个平面缩放到 pHash 使用的 `32 × 32`。

`ssim_score()` 使用高斯窗口计算局部均值、方差和协方差，形成 SSIM 图后求均值。
默认窗口边长 7，实际取 `min(win, h, w)`；不足 3 时返回 `None`。
SSIM 公式及默认 `data_range = 3.7` 没有改变。

### 4.2 已提交的变化

- 高斯核从只按窗口尺寸缓存，改为按窗口尺寸和设备缓存，容量从 8 增至 64。
- 高斯核在 CPU 构造并复制到目标设备；首次准备 NPU 常量时同步当前 stream。
- 灰度平面构造和 SSIM 入口明确转换为 FP32，并 detach 输入。
- 入口增加设备一致性检查：设备不同则返回 `None`。

当前核对到的代码仍为：

```python
if a.shape != b.shape or a.device != b.device:
    return None
```

最后的 `float(ssim_map.mean().item())` 是沿用的行为；在 NPU 上仍有标量回传同步。
当前实现没有将所有候选 SSIM 改造成一次设备端批量计算和统一回传。

### 4.3 为什么当前调用链通常同设备

tensor store 的普通和临时接收缓冲区都使用 `current_resized_tensor.device`。
候选从接收缓冲区经过 `view()`、`reshape()` 和 `clone()` 后仍保留该设备；
灰度平面构造也不改变设备。因此正常路径中查询与候选会位于同一设备。

候选最初从哪个设备写入 Memcache，不决定其读取目标；读取目标由当前查询决定。

但“正常路径同设备”不等于“异设备图片不具备可比性”。
静默返回 `None` 会跳过本来可以通过设备搬运完成比较的候选，并可能掩盖上层放置问题。

### 4.4 已确认但尚未落地的要求：允许异设备输入

用户已明确要求，为兼容异常或扩展调用场景，允许异设备输入进入 `ssim_score()`。
采用以下语义：

1. 以查询 `a.device` 为计算设备。
2. 保留形状检查及窗口尺寸检查，避免对无法评分的输入做无用搬运。
3. 将 `a` 转为 FP32，将 `b` 转为 `a.device` 上的 FP32。
4. 不原地修改调用方输入；设备复制失败时传播异常，不静默记为相似度未命中。

待应用的核心补丁：

```diff
-    if a.shape != b.shape or a.device != b.device:
+    if a.shape != b.shape:
         return None

     a = a.detach().to(dtype=torch.float32)
-    b = b.detach().to(dtype=torch.float32)
+    b = b.detach().to(device=a.device, dtype=torch.float32)
```

上面省略了保持不变的窗口和稳定常数计算代码。
这将尝试支持 CPU↔NPU 和不同 NPU 之间的输入对齐，具体设备组合仍受 PyTorch/torch_npu
复制能力约束。同设备且 dtype 已匹配时，`.to()` 通常不需要额外复制。

异设备路径会增加搬运与可能的同步成本。正常的 Memcache 加载路径仍应直接读到查询设备。
本次讨论中的生产代码补丁和相关回归测试尚未写入目标 worktree，不能标记为已完成或已验证。

## 5. perceptual_ssim_cache_matcher.py：检索优化主体

源码：[上游 matcher][up-matcher]、[优化版本 matcher][new-matcher]。

### 5.1 上游全表检索与零距离直查

上游复制 `phash_to_resized_mm_hashes` 的条目，排序后逐个计算汉明距离。
即使 `max_hamming == 0` 也会扫描。

当前零距离查询使用：

```python
resized_keys = self.phash_to_resized_mm_hashes.get(query_phash)
return list(resized_keys) if resized_keys else None
```

理由是 `hamming(a, b) == 0` 等价于 `a == b`。
若不同 pHash 数为 N、返回 key 数为 K，上游单次查询含全表排序与扫描；
当前零距离路径平均为 `O(1 + K)`，其中 K 来自列表复制。
返回副本避免暴露内部列表。零距离模式不填充 band 索引。

这里的“精确”仅指 pHash 值相同，仍须执行布局门控和 SSIM。

### 5.2 非零距离使用八段索引

新增索引：

```python
_phash_band_index: dict[tuple[int, int], set[int]]
```

含义为 `(段号, 该段的值) → 完整 pHash 集合`。64 位分为 8 段，每段 8 位。
查询自身的 8 个桶，合并完整 pHash，再计算完整汉明距离过滤假候选。

不漏召回的条件是 `max_hamming <= 7`：如果全部 8 段都不同，每段至少包含一个
不同 bit，总汉明距离至少为 8；反过来，距离最多为 7 时至少有一段相同。

配置原本就限制 `0 <= max_hamming < 8`，默认值仍为 5。
如果将来允许距离 8 或更大，必须重新设计分段策略或增加可靠的回退路径。

该保证针对满足汉明距离条件的 hash，不代表召回所有人眼认为相似的图片。
共享一个段也不足以判定完整距离合格，后续完整汉明过滤不可删除。

### 5.3 顺序与批内复用

桶结果使用集合去重后，按完整 pHash 数值排序，再展开其已有 resized key 列表。
这保留了上游的展开顺序。它不代表候选按汉明距离由小到大排列。

稳定顺序会影响后续分批及早停，因而属于行为兼容的一部分。

批量预热新增本次调用内的 `matches_by_phash`：对于 `[P, Q, P, P, Q]`，
只执行 P、Q 两次查询，然后按原请求顺序扩展结果，保留原有重复次数。
这不是跨请求生命周期的永久查询结果缓存。

### 5.4 原有状态锁扩展到新索引

`_similarity_state_lock` 是新版上游已有的。当前把 band 索引与完整映射、
`similarity_cache` 放到同一把锁中更新，避免读到部分发布的状态。

只有 resized tensor 成功持久化之后才发布索引；存储失败不发布。
新 pHash 首次加入时才填充八段索引，同一 pHash 后续只补充对应 key。

非零距离查询也在锁内收集并过滤候选。因此该实现不是无锁检索。
索引属于 matcher 实例，不新增跨进程共享、持久化或淘汰清理机制。

### 5.5 线程池与错误传播

上游主要按数量决定并行。当前还检查 pixel tensor 和 grid tensor 的设备：
有设备输入时保留在调用线程，只允许足够大的纯 CPU 预处理批次进入线程池。
原配置阈值为 3，最大 worker 数为 8，未在本次修改中调整。

原因是普通 worker 不能被假定继承调用线程的 NPU 当前设备和 producer stream。

CPU 并行从 `list(executor.map(...))` 改为提交 futures、`wait(futures)`、
再逐个 `future.result()`。重点在异常路径：等待所有已提交任务结束后再传播错误，
避免调用已经退出但后台仍继续持久化或更新索引。它不提供失败任务的事务回滚。

### 5.6 上游到底哪里是 >，哪里是 >=

以下位置均来自上游固定提交 `5fecef105`，不是根据优化版本反推：

| 文件与函数 | 上游行号 | 上游判断 | 用途 |
| --- | --- | --- | --- |
| matcher：`get_cached_similarity_match()` | [105][up-threshold-cache] | `score > threshold` | 读取已有 SSIM 缓存 |
| matcher：`find_and_cache_matched_ssim()` | [159][up-threshold-final] | `score >= threshold` | 新计算结果最终接受 |
| matcher：`_classify_similarity_cache_entries()` | [248][up-threshold-classify] | `score > threshold` | 预热分类 |
| matcher：`_get_ssim_score_list()` | [403][up-threshold-regular] | `regular_scores[0][0] > threshold` | 普通批次合格后跳过 oversized |
| tensor store：普通批次评分 | [298][up-threshold-store-regular] | `max_score > threshold` | 批次早停 |
| tensor store：oversized 批次评分 | [370][up-threshold-store-oversized] | `max_score > threshold` | 批次早停 |

因此上游混用 `>` 和 `>=`。当 `score` 恰好等于阈值时，新计算结果的最终判断
接受它，但缓存查询及预热不接受，也不会触发批次早停。

当前最终判断改为：

```python
if not score > self._config.ssim_threshold:
    return None
```

这统一为严格 `>`，改变了“刚好等于阈值”的最终接受行为；最终取出的分数为 NaN
时也会拒绝。不能据此声称对整个候选列表中的 NaN 做了全面清理或排序修复。

严格 `>` 是本次选择的阈值约定，不是 SSIM 公式的要求。若业务要求“达到阈值即接受”，
应同时修改上述缓存、分类、批次早停与最终判断为 `>=`，并补齐边界测试。
本次讨论没有批准或实施这个进一步调整。

## 6. memcache_resized_tensor_store.py：传输与评分生命周期

源码：[上游 tensor store][up-store]、[优化版本 tensor store][new-store]。

### 6.1 PUT 保留输入设备，显式选择传输方向

上游使用 `tensor.detach().to(device="cpu").contiguous()`，固定按 `COPY_H2G` 写入。
当前使用 `tensor.detach().contiguous()`，并根据实际设备选择方向：

| 设备 | PUT | GET |
| --- | --- | --- |
| CPU | `COPY_H2G` | `COPY_G2H` |
| NPU | `COPY_L2G` | `COPY_G2L` |

这避免 NPU 输入在 Python 层被强制先完整搬到 CPU。
缓存仍保留原张量 dtype；FP32 是 pHash/SSIM 的计算精度，不是修改存储格式。
传输适配器显式支持 CPU/NPU，其他设备会报错。

### 6.2 接收缓冲区跟随查询设备

上游常驻和临时缓冲区写死 `device="npu"`，当前使用查询张量的设备。
常驻缓冲区仍延迟分配；同设备复用，切换设备时同步、注销旧缓冲区并创建新缓冲区。
它不是每个设备同时保留一份的缓冲区池。

超过常驻容量的请求仍使用临时缓冲区。配置默认常驻容量 512 MiB，
单张量最大传输量 1 GiB，均为继承的限制。oversized 指超过常驻容量，
不意味着允许超过最大传输量。

### 6.3 PUT 锁保护注册、传输、注销全程

新增实例级 `_put_lock` 覆盖完整区间：

```text
register_buffer → put_from → unregister_buffer
```

backend 的 `register_buffer()` 使用共享 `_pending_buffers`，并发 PUT 可能互相
覆盖待注册状态；重复指针还可能出现一个任务提前注销另一个任务仍在使用的注册。
因此仅锁住 register 调用不够，必须保护整个生命周期，并用 `finally` 清理。

该锁会串行化同一 store 实例的 PUT 区间，不等价于整个 backend 的全局线程安全保证。

### 6.4 NPU 同步保护外部 SDK 指针访问和缓冲区复用

PyTorch 操作可能异步排队，外部 Memcache SDK 不会自动遵循 PyTorch producer stream。
当前增加以下同步：

- PUT 前：等张量生产或 contiguous 拷贝完成，再让 SDK 读取地址。
- GET 前：完成当前 stream 上对缓冲区的前置操作，再允许 SDK 写入。
- GET 后的 clone 完成前：不允许下一批覆盖或注销同一缓冲区。
- 设备切换和 shutdown 前：完成相关操作后再注销缓冲区。

上游已存在 clone，本次补充的是异步 clone 与下一次 SDK 覆盖之间的顺序保护：

```text
SDK 写入 A → 发起 clone(A) → 等 clone 完成 → SDK 才能用同一缓冲区写入 B
```

clone 后同步放在 `finally` 中，也覆盖处理中途异常的路径。
这不是零拷贝，也不是完全异步。同步针对当前调用线程的 stream，真实 SDK 完成语义、
非默认流和设备复制仍需要硬件集成验证。

### 6.5 空分数列表与结果排序修正

上游先 `max(batch_scores)` 再判断列表是否为空，整批读取失败或无法评分时会先抛异常。
当前把非空判断放在 `max()` 之前，利用短路求值保护空列表，允许继续后续批次。

每批评分后对累计结果执行：

```python
calculated_scores.sort(key=lambda item: (-item[0], item[1]))
```

即分数降序、同分按候选 `mm_hash` 字符串升序。
例如 A 为 0.98、B 为 0.995、阈值为 0.99，返回顺序必须为 B、A，
才能让上层读取第一项的判断与“该批已有合格候选”一致。

普通和 oversized 两种路径都应用了空列表保护和累计排序。

### 6.6 仍按批次早停，不保证全局最高 SSIM

找到第一个包含合格候选的批次后停止读取后续批次；普通批次合格后跳过 oversized。
当前保证在正常有限分数下选择已评分范围内的最佳项，不保证所有未加载候选中的最高分。

例如第一批已有 0.995，第二批可能存在 0.999，也不会为了寻找全局最大值继续读取。
这保留了上游用读取成本和计算成本换取较低检索延迟的策略。

SSIM 评分同样只在足够大的纯 CPU 批次使用线程池；设备计算留在调用线程，
CPU futures 全部完成后再传播错误。

### 6.7 新版接口属于继承内容

`ResizedTensorLoadRequest(output_index, resized_key, nbytes)` 已存在于 `5fecef105`。
当前继续使用其字段组织加载和评分，各批次重新编号本地输出位置。

同样继承的还有 model/grid/shape/dtype key、布局门控、容量分批与失败读取的跳过机制。
不能把它们全部归为此次检索优化新引入的功能。

## 7. 性能边界与验证状态

| 变化 | 可以从代码说明的收益或作用 | 不能直接宣称的结论 |
| --- | --- | --- |
| 零距离字典查询 | 不扫描全部 pHash，不计算汉明距离 | 整个查询严格 O(1)，因为仍要复制返回列表 |
| 八段索引 | 只对桶召回集合计算完整距离 | 固定加速倍数，或始终远小于全表的候选数 |
| 批内复用 | 重复 pHash 不重复检索 | 跨请求永久免检索 |
| 8 字节打包 | 缩小结果载荷，减少 Python 逐位循环 | 64 次同步变 1 次，或 8 倍速度提升 |
| 按设备计算与传输 | 减少强制整张量 CPU 搬移 | 消除所有 CPU/NPU 传输或同步 |
| 按批次排序早停 | 已评分范围内优选并减少后续加载 | 全部候选的全局最佳 SSIM |

非零距离索引增加约 8N 个集合成员关系，N 为不同 pHash 数量。
hash 分布集中时，桶候选仍可能接近全量。索引、锁、FP32 中间张量、接收缓冲区、
clone 和同步都应计入真实时延与内存评估。

此前 2026-10-22 的验证记录为：使用真实源码定义、模拟张量与外部 SDK 边界的
29 项控制流测试通过。该结果验证了检索顺序、阈值边界、空分数、请求映射及任务等待等，
不验证 DCT/SSIM 真实张量数值、NPU DMA 正确性或实际加速比。

当时环境缺少 PyTorch，完整 CPU 张量测试、NPU 和真实 Memcache 集成测试未运行。
`format.sh ci` 因缺少 pre-commit 未完成。本文没有重新执行这些测试，也没有将它们标为通过。

后续异设备 SSIM 变更需要覆盖 CPU→NPU、NPU→CPU、不同 NPU、同设备、输入不被修改、
形状或窗口不适配，以及设备复制失败的行为；硬件不足的用例应明确标记跳过。
性能对照应使用同配置下的 `5fecef105` 作为基线。

## 8. 复核固定提交的 PowerShell 命令

```powershell
$worktree = 'C:\Users\zongy\Downloads\vllm-ascend-phash-sync'
$base = '5fecef105a061c5692e2f727bf5eeed3297a224b'
$version = '44d109a9a3509da0c5ec89de8ec92d43f9492770'
$files = @(
    'vllm_ascend/distributed/ec_transfer/phash.py'
    'vllm_ascend/distributed/ec_transfer/ssim.py'
    'vllm_ascend/distributed/ec_transfer/perceptual_ssim_cache_matcher.py'
    'vllm_ascend/distributed/ec_transfer/memcache_resized_tensor_store.py'
)

# 上游与已提交优化版本之间的差异
git -C $worktree diff $base $version -- $files

# 单独查看当前工作目录相对已提交版本的差异
git -C $worktree diff $version -- $files

# 查看上游各处阈值判断
git -C $worktree grep -n -E 'score >|score >=|regular_scores\[0\]\[0\] >' $base -- $files
```

[up-phash]: https://github.com/hanxi-java/vllm-ascend/blob/5fecef105a061c5692e2f727bf5eeed3297a224b/vllm_ascend/distributed/ec_transfer/phash.py
[new-phash]: https://github.com/ZitWNG/vllm-ascend/blob/44d109a9a3509da0c5ec89de8ec92d43f9492770/vllm_ascend/distributed/ec_transfer/phash.py
[up-ssim]: https://github.com/hanxi-java/vllm-ascend/blob/5fecef105a061c5692e2f727bf5eeed3297a224b/vllm_ascend/distributed/ec_transfer/ssim.py
[new-ssim]: https://github.com/ZitWNG/vllm-ascend/blob/44d109a9a3509da0c5ec89de8ec92d43f9492770/vllm_ascend/distributed/ec_transfer/ssim.py
[up-matcher]: https://github.com/hanxi-java/vllm-ascend/blob/5fecef105a061c5692e2f727bf5eeed3297a224b/vllm_ascend/distributed/ec_transfer/perceptual_ssim_cache_matcher.py
[new-matcher]: https://github.com/ZitWNG/vllm-ascend/blob/44d109a9a3509da0c5ec89de8ec92d43f9492770/vllm_ascend/distributed/ec_transfer/perceptual_ssim_cache_matcher.py
[up-store]: https://github.com/hanxi-java/vllm-ascend/blob/5fecef105a061c5692e2f727bf5eeed3297a224b/vllm_ascend/distributed/ec_transfer/memcache_resized_tensor_store.py
[new-store]: https://github.com/ZitWNG/vllm-ascend/blob/44d109a9a3509da0c5ec89de8ec92d43f9492770/vllm_ascend/distributed/ec_transfer/memcache_resized_tensor_store.py
[up-threshold-cache]: https://github.com/hanxi-java/vllm-ascend/blob/5fecef105a061c5692e2f727bf5eeed3297a224b/vllm_ascend/distributed/ec_transfer/perceptual_ssim_cache_matcher.py#L105
[up-threshold-final]: https://github.com/hanxi-java/vllm-ascend/blob/5fecef105a061c5692e2f727bf5eeed3297a224b/vllm_ascend/distributed/ec_transfer/perceptual_ssim_cache_matcher.py#L159
[up-threshold-classify]: https://github.com/hanxi-java/vllm-ascend/blob/5fecef105a061c5692e2f727bf5eeed3297a224b/vllm_ascend/distributed/ec_transfer/perceptual_ssim_cache_matcher.py#L248
[up-threshold-regular]: https://github.com/hanxi-java/vllm-ascend/blob/5fecef105a061c5692e2f727bf5eeed3297a224b/vllm_ascend/distributed/ec_transfer/perceptual_ssim_cache_matcher.py#L403
[up-threshold-store-regular]: https://github.com/hanxi-java/vllm-ascend/blob/5fecef105a061c5692e2f727bf5eeed3297a224b/vllm_ascend/distributed/ec_transfer/memcache_resized_tensor_store.py#L298
[up-threshold-store-oversized]: https://github.com/hanxi-java/vllm-ascend/blob/5fecef105a061c5692e2f727bf5eeed3297a224b/vllm_ascend/distributed/ec_transfer/memcache_resized_tensor_store.py#L370
