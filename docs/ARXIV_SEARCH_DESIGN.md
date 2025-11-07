# arXiv 搜索机制调研与设计

## 1. 官方接口概览
- **主访问方式**：`http://export.arxiv.org/api/query`
- **协议**：基于 Atom Feed（可使用 HTTP GET）。
- **常用参数**：
  - `search_query`：支持布尔组合（`ti:` 标题，`abs:` 摘要，`au:` 作者，`cat:` 分类）。
  - `start`：起始偏移，默认 `0`。
  - `max_results`：每次返回数量，建议 <= 50 以遵守限流。
  - `sortBy` / `sortOrder`：排序方式（`relevance`, `lastUpdatedDate`, `submittedDate`）。
- **限流建议**：官方要求单 IP 每分钟不超过 15 次请求，需在客户端实现速率限制。

## 2. 查询生成策略
1. **基础 Prompt 模板**（供 SearchAgent 调用 LLM 生成检索式）：
   ```text
   你是一名检索专家。给定综述大纲中的小节标题与主题背景，请输出 1-3 个适合 arXiv API 的检索式。
   - 使用英文关键词。
   - 优先考虑题目、摘要字段。
   - 使用逻辑运算符提升相关性。
   - 如需时间过滤，可附加 `submittedDate` 范围建议。
   输出 JSON 数组，每项含 `query`, `rationale`, `expected_paper_types`。
   ```
2. **语义扩展**：利用同义词或技术术语，结合上下文关键词生成多轮查询。
3. **反馈循环**：
   - 若某次检索结果低于阈值，生成备用查询。
   - 将失败原因记录在缓存，避免重复请求。

## 3. 客户端实现建议
- 使用 `aiohttp` 或 `httpx.AsyncClient` 构建异步请求，便于限流与并发控制。
- 实现 `ArxivClient`：
  ```python
  class ArxivClient:
      def __init__(self, rate_limiter: RateLimiter, session: httpx.AsyncClient):
          ...

      async def query(self, search_query: str, *, start: int = 0, max_results: int = 20) -> ArxivResult:
          """返回解析后的论文列表，并缓存摘要"""
  ```
- 解析 Atom Feed 可使用 `feedparser`，将结果标准化为：
  ```python
  @dataclass
  class ArxivPaper:
      arxiv_id: str
      title: str
      summary: str
      authors: list[str]
      published: datetime
      updated: datetime
      pdf_url: str
      categories: list[str]
  ```
- **缓存策略**：
  - 使用 `functools.lru_cache` 或自定义缓存层（如 SQLite/Redis）。
  - 缓存 key：`arxiv_id`，存储摘要、元数据、获取时间。
  - 幻觉反馈智能体可直接读取缓存，避免重复请求。

## 4. 错误处理与重试
- 对 `429` 响应应用指数退避策略。
- 捕获网络异常并记录日志（包括查询语句、状态码、响应时间）。
- 若返回条目为空：
  - 通知协调智能体触发备用查询。
  - 在 `SearchTask` 中附带失败原因，供后续分析。

## 5. 安全与合规
- 遵守 arXiv 使用条款，不批量下载全文。
- 缓存中存储的摘要仅用于内部验证与撰写，不公开泄露。
- 对外输出引用时，仅提供 arXiv ID 与公共元数据。

