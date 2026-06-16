# LLM 流式接收支持

## TL;DR
> **Summary**: 为 LLM 请求添加 `stream=True` 流式接收支持，解决代理连接中途截断导致大请求丢失的问题。流式接收逐 chunk 获取内容，断连时保留已收到的部分，配合现有续写机制补全。
> **Deliverables**: 流式 API 调用、部分内容保留、断点续写集成
> **Effort**: Short
> **Parallel**: NO
> **Critical Path**: LLMProvider.stream → call_llm_with_transport_retry → 流式 chunk 收集 → 续写集成

## Context
### Original Request
用户通过代理连接 OPENCODE API 时，大请求/响应在代理层被截断，无法获取完整结果。需要流式接收机制，逐 chunk 接收内容，断连时保留已收到部分。

### Interview Summary
- 直连 OPENCODE API 有网络问题
- 代理连接会在中途截断
- 大请求无法一次完成
- 需要流式接收 + 断点续写

### Metis Review
- 流式接收需要正确处理 `finish_reason`，确保续写机制能触发
- 需要兼容非流式模式（向后兼容）
- 流式模式下的 token usage 需要从最后一个 chunk 提取（`stream_options={"include_usage": True}`）
- 流式中断应先传输重试，耗尽后才触发续写
- 流式只用于非工具调用场景（`tools` 参数存在时禁用流式）
- 续写、reasoning followup、JSON fix 调用也会走流式（通过 `call_llm` → `call_llm_with_transport_retry`），这是预期行为

## Work Objectives
### Core Objective
实现在 `call_llm_with_transport_retry` 层的流式 API 调用，在代理截断时保留已收到的部分内容，配合续写机制补全。

### Deliverables
1. `LLMProvider.stream` 字段
2. `call_llm_with_transport_retry` 流式支持（逐 chunk 收集，中断时传输重试）
3. 流式中断且传输重试耗尽后返回 `finish_reason="length"` 触发续写
4. 配置文件添加 `stream` 选项

### Definition of Done
- 流式模式下，代理截断时能保留已收到的部分内容
- 流式中断先触发传输重试，耗尽后才触发续写
- 非流式模式行为不变（向后兼容）
- 工具调用不走流式
- 所有现有测试通过

### Must Have
- `stream: true` 配置选项
- 流式 chunk 收集（在 `call_llm_with_transport_retry` 层）
- `stream_options={"include_usage": True}` 获取 usage
- 流式中断 → 传输重试 → 续写
- 工具调用不走流式

### Must NOT Have
- 不在 `_call_llm_raw` 中添加流式逻辑
- 不改变非流式模式的行为
- 不改变续写逻辑本身
- 不破坏现有测试

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: tests-after
- QA policy: Every task has agent-executed scenarios
- Evidence: .omo/evidence/task-{N}-{slug}.{ext}

## Execution Strategy
### Parallel Execution Waves
Wave 1: LLMProvider.stream + config 读取
Wave 2: call_llm_with_transport_retry 流式支持
Wave 3: 配置文件更新 + 测试

### Dependency Matrix
Task 1 → Task 2 → Task 3

## TODOs

- [ ] 1. LLMProvider 添加 stream 字段 + config 读取

  **What to do**:
  - 在 `LLMProvider` dataclass 添加 `stream: bool = False` 字段（在 `proxy` 字段之后）
  - 在 `_resolve_provider_from_config` 读取 `stream` 配置
  - 在 `LLMProvider` 构造函数传入 `stream=stream`

  **Must NOT do**: 不改变其他字段

  **Recommended Agent Profile**:
  - Category: `quick` - 简单字段添加
  - Skills: [] - 无需特殊技能

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: [2] | Blocked By: []

  **References**:
  - Pattern: `dd_clip_miner_llm/llm.py:44-61` - LLMProvider dataclass
  - Pattern: `dd_clip_miner_llm/llm.py:122-143` - _resolve_provider_from_config return

  **Acceptance Criteria**:
  - [ ] LLMProvider 有 stream 字段，默认 False
  - [ ] config 中 `stream: true` 被正确读取

  **QA Scenarios**:
  ```
  Scenario: stream 配置读取
    Tool: Bash
    Steps: python -c "from dd_clip_miner_llm.config import load_config; from dd_clip_miner_llm.llm import build_providers; c = load_config('config.yaml'); ps = build_providers(c); print(ps[0].stream)"
    Expected: False (默认值)
    Evidence: .omo/evidence/task-1-stream-config.txt
  ```

  **Commit**: YES | Message: `feat(llm): add stream field to LLMProvider` | Files: [dd_clip_miner_llm/llm.py]

- [ ] 2. call_llm_with_transport_retry 流式支持

  **What to do**:
  - 在 `call_llm_with_transport_retry` 中，检测 `provider.stream` 且无 `tools`
  - 当流式启用时，在 `_build_request_kwargs` 返回 kwargs 后手动添加：
    - `kwargs["stream"] = True`
    - `kwargs["stream_options"] = {"include_usage": True}`
  - 调用 `_call_llm_raw(client, kwargs)` 获取流式迭代器
  - 逐 chunk 收集内容，构建 `SimpleNamespace` 响应对象（需包含以下字段）：
    - `response.choices[0].message.content` — 收集的完整内容
    - `response.choices[0].message.reasoning_content` — `""`
    - `response.choices[0].message.tool_calls` — `None`
    - `response.choices[0].finish_reason` — `"stop"` 或 `"length"`
    - `response.usage` — 从最后一个 chunk 提取（需 `stream_options`）
    - `response.model` — 从最后一个 chunk 提取
  - 如果流式中断（异常），继续传输重试循环（不立即返回）
  - 传输重试耗尽后，如果有部分内容，构建响应对象（`finish_reason="length"`）并返回（不 raise）
  - 正常完成时，从最后一个 chunk 提取 usage，构建完整响应对象

  **Must NOT do**:
  - 不在 `_call_llm_raw` 中添加流式逻辑
  - 工具调用时不走流式（`tools` 参数存在时禁用流式）
  - 不改变非流式模式的行为

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - 需要理解 OpenAI 流式 API
  - Skills: [] - 无需特殊技能

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: [3] | Blocked By: [1]

  **References**:
  - Pattern: `dd_clip_miner_llm/llm.py:255-257` - 当前 _call_llm_raw（不改动）
  - Pattern: `dd_clip_miner_llm/llm.py:270-345` - 当前 call_llm_with_transport_retry
  - API: OpenAI streaming - `stream=True` 返回迭代器，每个 chunk 有 `choices[0].delta.content`
  - API: `stream_options={"include_usage": True}` - 最后一个 chunk 包含 usage

  **Acceptance Criteria**:
  - [ ] `stream=True` 且无 tools 时，逐 chunk 收集内容
  - [ ] 流式中断时，继续传输重试（不立即触发续写）
  - [ ] 传输重试耗尽且有部分内容时，返回 `finish_reason="length"`
  - [ ] 正常完成时返回完整响应对象（含 usage）
  - [ ] 工具调用时不走流式
  - [ ] 非流式模式行为不变

  **QA Scenarios**:
  ```
  Scenario: 流式正常完成
    Tool: Bash
    Steps: mock 流式 API，验证完整响应被正确收集
    Expected: 返回完整 content + finish_reason="stop" + usage
    Evidence: .omo/evidence/task-2-stream-complete.txt

  Scenario: 流式中断触发传输重试
    Tool: Bash
    Steps: mock 流式 API 在第 3 个 chunk 后抛出异常，第二次成功
    Expected: 传输重试成功，返回完整内容
    Evidence: .omo/evidence/task-2-stream-retry.txt

  Scenario: 流式中断且传输重试耗尽
    Tool: Bash
    Steps: mock 流式 API 持续抛出异常
    Expected: 返回部分内容 + finish_reason="length"
    Evidence: .omo/evidence/task-2-stream-partial.txt

  Scenario: 工具调用不走流式
    Tool: Bash
    Steps: mock API，传入 tools 参数，验证 stream=True 未被设置
    Expected: kwargs 中无 stream=True
    Evidence: .omo/evidence/task-2-stream-no-tools.txt
  ```

  **Commit**: YES | Message: `feat(llm): add streaming support to call_llm_with_transport_retry` | Files: [dd_clip_miner_llm/llm.py]

- [ ] 3. 配置文件更新 + 测试

  **What to do**:
  - 更新 config.yaml、config.example.yaml、config.deepseek.example.yaml、config.daily-summary.example.yaml 添加 `stream` 选项
  - 更新 README 文档
  - 添加流式测试
  - 运行完整测试套件

  **Must NOT do**: 不破坏现有测试

  **Recommended Agent Profile**:
  - Category: `quick` - 配置文件更新
  - Skills: [] - 无需特殊技能

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: [] | Blocked By: [2]

  **References**:
  - Pattern: `config.example.yaml:84-117` - provider 配置格式
  - Pattern: `tests/test_llm_retry.py` - 现有测试模式

  **Acceptance Criteria**:
  - [ ] 所有 config 文件有 `# stream: false` 注释
  - [ ] README 有流式模式说明
  - [ ] 流式测试通过
  - [ ] 所有现有测试通过（211 passed）

  **QA Scenarios**:
  ```
  Scenario: 配置文件格式正确
    Tool: Bash
    Steps: python -c "from dd_clip_miner_llm.config import load_config; load_config('config.yaml')"
    Expected: 无报错
    Evidence: .omo/evidence/task-3-config-valid.txt

  Scenario: 完整测试套件
    Tool: Bash
    Steps: python -m pytest tests/ -q --tb=short
    Expected: 211+ passed, 0 failed
    Evidence: .omo/evidence/task-3-test-results.txt
  ```

  **Commit**: YES | Message: `docs: add stream option to config files` | Files: [config*.yaml, README.md, tests/test_llm_retry.py]

## Final Verification Wave
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
每个 task 单独 commit，message 格式: `type(scope): desc`

## Success Criteria
- 流式模式下，代理截断时能保留已收到的部分内容
- 流式中断先触发传输重试，耗尽后才触发续写
- 非流式模式行为不变
- 211+ 测试通过
