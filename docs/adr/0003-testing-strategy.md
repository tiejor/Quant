# 测试策略：临时 DuckDB 而非 mock

单元测试不 mock DuckDB 连接，使用临时 DuckDB 文件精确控制测试数据。冒烟测试使用生产库（`data/quant.duckdb`）。

**理由：** mock 数据库连接/查询返回值需要精确复制 DuckDB 的返回格式（DataFrame、dtype、Index），维护成本高且容易和真实行为漂移。临时 DuckDB 文件创建和销毁成本极低（内存文件或几十行数据），测试更接近生产行为，且无需维护 mock 替身。冒烟测试直接用生产库验证"真实数据能跑通"，是最后一道防线。

**放弃的方案：**
- Mock/unittest.patch：需要伪造 `duckdb.connect()` 返回值，维护假 DataFrame 的格式一致性，mock 和真实 DuckDB 之间的行为偏差曾导致测试通过但运行崩溃
- 纯快照测试（无单元测试）：无法精确控制边界条件（ST 当天、上市 364 vs 365 天），只能测常见路径

**后果：** 测试需要 DuckDB 作为测试依赖（`pip install duckdb`）。临时文件存放在 `tests/data/` 目录下，测试结束自动清理。大表并行拉取（joblib）不在单元测试范围，留给集成/冒烟层。
