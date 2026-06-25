# agent-demo

一个最小 LangChain agent 示例，包含 ReAct、Plan-and-Resolve 和 Reflection 三种模式。

## 环境

```bash
conda env create -f environment.yml
conda activate agent-demo
```

如果环境已经存在，可更新依赖：

```bash
conda env update -f environment.yml --prune
```

## 配置

```bash
cp .env.example .env
```

然后在 `.env` 中填写：

```bash
LLM_API_KEY=...
LLM_BASE_URL=...
LLM_MODEL_ID=...
```

## 运行

```bash
python react_agent.py
```

或传入自己的问题：

```bash
python react_agent.py "计算 18 * 23，并告诉我当前 UTC 时间"
```

运行 Plan-and-Resolve agent：

```bash
python plan_resolve_agent.py
```

或传入自己的问题：

```bash
python plan_resolve_agent.py "计算 18 * 23，并告诉我当前 UTC 时间"
```

运行 Reflection agent：

```bash
python reflection_agent.py
```

或传入自己的问题：

```bash
python reflection_agent.py "计算 18 * 23，并告诉我当前 UTC 时间"
```

也可以指定最多反思轮数：

```bash
python reflection_agent.py --max-reflections 3 "计算 18 * 23，并告诉我当前 UTC 时间"
```

## 结构

- `react_agent.py`: ReAct agent 主程序，包含 `calculator` 和 `current_time` 两个工具。
- `plan_resolve_agent.py`: Plan-and-Resolve agent 主程序，先规划步骤，再逐步执行并汇总答案。
- `reflection_agent.py`: Reflection agent 主程序，先生成答案，再反思检查并按需修订。
- `environment.yml`: conda 环境定义。
- `.env.example`: 本地配置模板。
