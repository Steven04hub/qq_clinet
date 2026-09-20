# QQ AI Bot 开发规格书

## 1. 项目目标

开发一个基于普通 QQ 小号的 AI 群聊机器人。

QQ 小号通过 NapCatQQ 接入 QQ，Python 程序通过 OneBot 11 WebSocket 与 NapCat 通信。

第一阶段目标：

用户在指定 QQ 群中：

```text
@机器人 什么是 Transformer？
```

机器人能够：

```text
Transformer 是一种基于注意力机制的神经网络架构……
```

第一版要求稳定、结构清晰、可测试。

不要一开始实现复杂 Agent、RAG、MCP、管理后台、数据库或 Docker。

---

# 2. 总体架构

```text
                   QQ 群
                     │
                     ▼
                普通 QQ 小号
                     │
                     ▼
                  NapCat
                     │
              OneBot 11 WS
                     │
                     ▼
              OneBotClient
                     │
             ┌───────┴───────┐
             ▼               ▼
       Event Dispatcher   Action Manager
             │               │
             ▼               │
       Message Router        │
             │               │
             ▼               │
         ChatHandler         │
             │               │
        ┌────┴────┐          │
        ▼         ▼          │
 SessionStore  RateLimiter   │
        │                    │
        ▼                    │
    LLMService               │
        │                    │
        ▼                    │
 DeepSeek/OpenAI             │
        │                    │
        └──────────► send_group_msg
```

Python Bot 是独立程序。

NapCat 只负责：

- QQ 登录
- QQ 消息接收
- QQ 消息发送
- OneBot 协议转换

业务逻辑全部放在 Python 项目中。

---

# 3. 技术栈

使用：

```text
Python 3.12
uv
asyncio
websockets
pydantic
pydantic-settings
openai
pytest
pytest-asyncio
```

暂时不要使用：

```text
NoneBot
FastAPI
Redis
PostgreSQL
SQLAlchemy
Celery
RabbitMQ
Docker
LangChain
LangGraph
MCP
RAG
```

以后有实际需求再加入。

---

# 4. Python 项目目录

创建：

```text
qq-ai-bot/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   │
│   ├── onebot/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── actions.py
│   │   ├── events.py
│   │   ├── messages.py
│   │   └── models.py
│   │
│   ├── handlers/
│   │   ├── __init__.py
│   │   ├── router.py
│   │   ├── chat.py
│   │   └── commands.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm.py
│   │   ├── session.py
│   │   ├── rate_limit.py
│   │   └── dedup.py
│   │
│   └── utils/
│       └── logging.py
│
├── tests/
│   ├── test_message_parser.py
│   ├── test_router.py
│   ├── test_session.py
│   ├── test_rate_limit.py
│   └── test_onebot_client.py
│
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── SPEC.md
```

保持模块职责单一。

不要为了所谓“企业级架构”继续增加无意义层级。

---

# 5. NapCat 接入方式

第一版选择：

```text
NapCat = WebSocket Server

Python = WebSocket Client
```

连接：

```text
ws://127.0.0.1:3001
```

NapCat 推荐配置：

```text
WebSocket Server
host = 127.0.0.1
port = 3001
messagePostFormat = array
reportSelfMessage = false
token = 用户自己配置
```

不要监听公网地址。

开发环境只允许本机连接。

Python 程序必须支持：

1. 建立 WebSocket
2. 断线自动重连
3. 接收 OneBot Event
4. 发送 OneBot Action
5. 使用 echo 匹配 Action Response
6. 请求超时
7. graceful shutdown

---

# 6. OneBotClient 设计

实现：

```python
class OneBotClient:
    async def connect(self) -> None:
        ...

    async def run(self) -> None:
        ...

    async def close(self) -> None:
        ...

    async def call_action(
        self,
        action: str,
        params: dict,
        timeout: float = 10.0,
    ) -> dict:
        ...

    async def send_group_message(
        self,
        group_id: int,
        message: list[dict] | str,
    ) -> int:
        ...
```

---

## 6.1 WebSocket 接收循环

一个 WebSocket 中可能同时收到：

### Event

例如：

```json
{
    "post_type": "message",
    "message_type": "group",
    "group_id": 123456,
    "user_id": 654321
}
```

### Action Response

例如：

```json
{
    "status": "ok",
    "retcode": 0,
    "data": {},
    "echo": "xxxxxxxx"
}
```

因此收到 JSON 后必须进行分流：

```text
收到 JSON
   │
   ├── 存在 echo 并匹配 pending request
   │       ↓
   │    Action Response
   │
   └── 存在 post_type
           ↓
         Event
```

禁止：

```text
发送 Action
↓
直接 ws.recv()
↓
认为下一条一定是 Action Response
```

因为 Action Response 和 QQ Event 可能交错到达。

---

# 7. Action Request Manager

调用 OneBot Action 时：

生成唯一：

```text
echo = UUID
```

维护：

```python
pending_requests: dict[str, asyncio.Future]
```

流程：

```text
call_action()
    │
    ├─ create Future
    ├─ pending[echo] = Future
    ├─ ws.send(...)
    │
    └─ await Future
```

收到：

```json
{
    "echo": "abc"
}
```

以后：

```python
future = pending.pop("abc")
future.set_result(response)
```

必须实现 timeout。

例如：

```text
10 秒没有 Action Response
→ raise OneBotTimeoutError
```

连接断开以后：

```text
所有 pending Future
→ 设置 ConnectionError
→ 清空 pending_requests
```

---

# 8. WebSocket 重连

实现无限重连：

```text
连接
 │
 ├── 成功
 │     ↓
 │   recv loop
 │
 └── 断开
       ↓
    等待 backoff
       ↓
      重连
```

backoff：

```text
1s
2s
4s
8s
15s
15s
...
```

成功连接以后重新归零。

关闭程序时不得继续重连。

---

# 9. Event Models

使用 Pydantic 建模最关键字段。

不要试图完整建模 OneBot 所有事件。

只做当前需要的事件。

例如：

```python
class GroupSender(BaseModel):
    user_id: int
    nickname: str = ""
    card: str = ""
    role: str = "member"


class GroupMessageEvent(BaseModel):
    post_type: str
    message_type: str
    message_id: int
    group_id: int
    user_id: int
    self_id: int
    message: list[MessageSegment]
    raw_message: str = ""
    sender: GroupSender
```

未知字段：

```python
model_config = ConfigDict(extra="ignore")
```

保持对 NapCat 扩展字段兼容。

---

# 10. Message Segment

至少支持：

```text
text
at
reply
image
face
```

第一版 AI 只理解：

```text
text
at
reply
```

其它 Segment 可以忽略，但不能导致解析崩溃。

定义：

```python
class MessageSegment(BaseModel):
    type: str
    data: dict
```

暂时不需要为每种 Segment 创建十几个 Class。

---

# 11. @机器人识别

必须读取：

```text
event.self_id
```

机器人 QQ：

```text
self_id
```

检测 message segments 是否存在：

```json
{
    "type": "at",
    "data": {
        "qq": "<self_id>"
    }
}
```

不要使用 nickname 判断。

不要使用字符串：

```text
@机器人名字
```

判断。

QQ 昵称可以修改，不可靠。

---

# 12. 文本提取

输入：

```text
[
    AT(bot),
    TEXT(" 什么是"),
    TEXT(" Transformer？")
]
```

输出：

```text
什么是 Transformer？
```

删除：

```text
@机器人
reply segment
```

保留：

```text
text
```

多个 text segment 合并。

做：

```python
def extract_plain_text(
    segments: list[MessageSegment],
    bot_id: int,
) -> str:
    ...
```

并写单元测试。

---

# 13. Message Router

Router 只负责决定：

```text
这个 Event 应该交给谁？
```

不要在 Router 中调用 LLM。

流程：

```text
GroupMessageEvent
      │
      ├── 非群消息
      │      → ignore
      │
      ├── sender == bot
      │      → ignore
      │
      ├── 不在 allowed groups
      │      → ignore
      │
      ├── command
      │      → CommandHandler
      │
      ├── @bot
      │      → ChatHandler
      │
      └── otherwise
             → ignore
```

---

# 14. 防止机器人回复自己

必须同时做两层保护。

### 第一层

NapCat：

```text
reportSelfMessage = false
```

### 第二层

Python：

```python
if event.user_id == event.self_id:
    return
```

即使 NapCat 配置被修改，也不会出现机器人无限回复自己的情况。

---

# 15. 群白名单

第一版必须使用群白名单。

环境变量：

```text
ALLOWED_GROUP_IDS=123456789,987654321
```

行为：

```text
group_id in whitelist
→ 可以处理

否则
→ 完全忽略
```

禁止第一版默认响应机器人加入的所有 QQ 群。

---

# 16. ChatHandler

职责：

```text
收到已经通过 Router 检查的聊天请求
        ↓
检查 dedup
        ↓
检查 rate limit
        ↓
取得 session
        ↓
调用 LLM
        ↓
更新 session
        ↓
回复 QQ
```

定义：

```python
class ChatHandler:
    async def handle(
        self,
        event: GroupMessageEvent,
    ) -> None:
        ...
```

---

# 17. Session 设计

第一版使用内存。

暂时不要 Redis。

Session Key：

```text
group:{group_id}
```

也就是：

```text
一个QQ群共享一个上下文
```

而不是：

```text
group_id:user_id
```

原因：

QQ 群对话本身是多人连续对话。

例如：

```text
张三：
@Bot Python 和 Java 有什么区别？

Bot：
...

李四：
@Bot 那性能呢？
```

第二句话应该可以理解上一轮。

---

# 18. Session 内容

不要把 QQ 群所有聊天都存入上下文。

只保存：

```text
明确触发机器人的消息
+
机器人的回答
```

结构：

```python
[
    {
        "role": "user",
        "content": "[张三/123456] 什么是 Transformer？"
    },
    {
        "role": "assistant",
        "content": "..."
    }
]
```

保存昵称是为了多人群聊时让 LLM 知道当前说话者是谁。

---

# 19. SessionStore 接口

虽然第一版使用内存，但是定义接口：

```python
class SessionStore(Protocol):
    async def get(self, session_id: str) -> list[dict]:
        ...

    async def append(
        self,
        session_id: str,
        message: dict,
    ) -> None:
        ...

    async def clear(self, session_id: str) -> None:
        ...
```

实现：

```python
class MemorySessionStore:
    ...
```

以后替换 Redis 时，ChatHandler 不需要修改。

---

# 20. Context 限制

设置：

```text
MAX_CONTEXT_MESSAGES=20
```

只保留最近：

```text
20 条 message
```

注意：

```text
user + assistant = 2 条
```

所以大约对应 10 轮。

第一版不需要复杂 token counting。

---

# 21. 每群串行处理

群上下文存在竞争问题。

例如同时：

```text
A：@bot 问题1
B：@bot 问题2
```

如果同时调用 LLM：

```text
context 顺序可能混乱
```

因此：

```python
group_locks: dict[int, asyncio.Lock]
```

同一个群：

```text
串行
```

不同群：

```text
并行
```

也就是：

```text
Group A ───── sequence

Group B ───── sequence

A 和 B 可以同时执行
```

---

# 22. 全局并发限制

LLM 使用：

```python
asyncio.Semaphore
```

例如：

```text
MAX_LLM_CONCURRENCY=3
```

调用：

```python
async with llm_semaphore:
    response = await llm.chat(...)
```

避免一个 QQ 群瞬间制造几十个 API 请求。

---

# 23. Rate Limiter

第一版使用内存。

至少实现：

```text
同一个用户：
5 秒内最多请求一次
```

配置：

```text
USER_RATE_LIMIT_SECONDS=5
```

触发频率限制时：

第一版直接忽略。

不要回复：

```text
你请求太快了
```

否则恶意刷请求仍然会导致机器人刷屏。

---

# 24. Dedup

维护：

```text
最近处理过的 message_id
```

避免：

- reconnect
- event 重复推送
- 程序内部异常

导致一次 QQ 消息调用两次 LLM。

实现：

```python
class MessageDeduplicator:
    def seen(self, message_id: int) -> bool:
        ...
```

最多保存例如：

```text
1000 个 message_id
```

超过以后淘汰最旧项。

---

# 25. LLM Service

不要在 handler 中直接使用 OpenAI SDK。

定义统一接口：

```python
class LLMService:
    async def chat(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        ...
```

实现：

```python
class OpenAICompatibleLLM:
    ...
```

配置：

```text
LLM_API_KEY
LLM_BASE_URL
LLM_MODEL
```

因此可以兼容：

```text
DeepSeek
OpenAI
其它 OpenAI-compatible API
```

业务层不知道具体使用哪个厂商。

---

# 26. System Prompt

单独定义配置。

默认：

```text
你是QQ群中的AI助手。

回答用户问题时：
1. 使用自然、简洁的中文。
2. 优先直接回答问题。
3. 不要每次都进行自我介绍。
4. 群聊环境中避免不必要的超长回复。
5. 上下文中可能存在多个不同用户，请根据昵称区分说话者。
```

不要把 System Prompt 硬编码进 ChatHandler。

---

# 27. LLM 异常处理

分别处理：

```text
timeout
HTTP/API error
rate limit
empty response
unknown exception
```

用户侧只回复简短信息，例如：

```text
模型暂时不可用，请稍后再试。
```

日志中保存真正异常。

不要把：

```text
API Key
Authorization header
完整 stack trace
```

发送到 QQ。

---

# 28. 回复方式

优先使用：

```text
reply segment
+
text segment
```

例如：

```json
[
    {
        "type": "reply",
        "data": {
            "id": "原消息ID"
        }
    },
    {
        "type": "text",
        "data": {
            "text": "回答内容"
        }
    }
]
```

这样群里可以明确知道机器人回答的是谁。

---

# 29. 长回答

定义应用级限制：

```text
MAX_REPLY_CHARS=1500
```

如果模型输出超过限制：

按：

```text
段落
↓
句号
↓
硬切
```

拆成多个消息。

不要直接：

```text
answer[:1500]
```

导致句子截断。

第一版最多发送：

```text
3 段
```

超过部分：

```text
截断
```

避免机器人刷屏。

---

# 30. Commands

第一版只实现：

```text
/help
/clear
/status
```

## /help

返回：

```text
@机器人 + 问题：和 AI 对话
/clear：清除当前群上下文
/status：查看机器人状态
```

## /clear

清除：

```text
当前 group_id session
```

## /status

返回简单状态：

```text
Bot online
Model: xxx
Context: 8/20
```

---

# 31. Admin

配置：

```text
ADMIN_QQ_IDS
```

未来管理命令才能使用。

第一版 `/help`、`/clear`、`/status` 不要求管理员。

暂时不要实现：

```text
kick
mute
delete message
group admin
```

机器人第一版没有必要拥有群管理操作能力。

---

# 32. Config

使用：

```python
pydantic_settings.BaseSettings
```

例如：

```python
class Settings(BaseSettings):
    napcat_ws_url: str = "ws://127.0.0.1:3001"
    napcat_access_token: str

    allowed_group_ids: list[int]
    admin_qq_ids: list[int] = []

    llm_api_key: str
    llm_base_url: str
    llm_model: str

    max_context_messages: int = 20
    max_llm_concurrency: int = 3
    user_rate_limit_seconds: int = 5
    max_reply_chars: int = 1500
```

配置解析要有测试。

---

# 33. .env.example

创建：

```text
NAPCAT_WS_URL=ws://127.0.0.1:3001
NAPCAT_ACCESS_TOKEN=change-me

ALLOWED_GROUP_IDS=
ADMIN_QQ_IDS=

LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL=

MAX_CONTEXT_MESSAGES=20
MAX_LLM_CONCURRENCY=3
USER_RATE_LIMIT_SECONDS=5
MAX_REPLY_CHARS=1500
```

`.env`：

必须加入：

```text
.gitignore
```

禁止提交 API Key。

---

# 34. Logging

使用 Python 标准：

```python
logging
```

不要额外加入 loguru。

日志至少包括：

```text
NapCat connected
NapCat disconnected
NapCat reconnecting

group_id
user_id
message_id

LLM request started
LLM request finished
LLM latency

Action failed
LLM failed
```

不要默认打印：

```text
完整 LLM API Key
NapCat token
Authorization
```

用户消息可以 DEBUG 级记录，但 INFO 不打印完整聊天内容。

---

# 35. main.py

main.py 只负责：

```text
load settings
↓
configure logging
↓
construct dependencies
↓
start bot
↓
handle shutdown
```

不能把业务逻辑全部写进 main.py。

大致：

```python
async def main():
    settings = get_settings()

    session_store = MemorySessionStore(...)
    llm = OpenAICompatibleLLM(...)
    router = MessageRouter(...)
    client = OneBotClient(...)

    await client.run(router.handle_event)
```

---

# 36. graceful shutdown

处理：

```text
SIGINT
SIGTERM
Ctrl+C
```

关闭：

```text
停止 reconnect
↓
关闭 WebSocket
↓
cancel background tasks
↓
cancel pending action futures
↓
退出
```

不得产生大量：

```text
Task was destroyed but it is pending
```

---

# 37. 测试策略

不要连接真正 QQ 才测试所有东西。

大部分逻辑必须使用 unit test。

---

## test_message_parser

测试：

```text
AT bot + text
multiple text segments
at another person
reply + text
image + text
empty text
```

---

## test_router

测试：

```text
非群消息 → ignore

机器人自己消息 → ignore

非白名单群 → ignore

没有 @bot → ignore

@bot → ChatHandler

/clear → CommandHandler
```

---

## test_session

测试：

```text
append
get
clear
max context
group isolation
```

---

## test_rate_limit

测试：

```text
first request allowed
immediate second denied
after timeout allowed
different users independent
```

---

## test_onebot_client

Mock WebSocket。

测试：

```text
action → echo

不同 echo 对应不同 Future

event 与 action response 交错

request timeout

disconnect cancels pending requests
```

尤其必须测试：

```text
send action A
↓
先收到 QQ Event B
↓
后收到 Response A
```

程序仍能正确处理。

这是 OneBotClient 最重要的测试之一。

---

# 38. 第一阶段验收标准

Phase 1：

```text
NapCat
↓
WebSocket
↓
Python
```

必须做到：

- Python 可以连接 NapCat。
- 可以打印群消息 event。
- 断开以后可以重新连接。
- 可以调用 send_group_msg。
- echo request-response 正常工作。
- pytest 全部通过。

此阶段禁止调用 LLM。

---

# 39. 第二阶段验收标准

Phase 2：

实现：

```text
@bot echo hello
```

机器人返回：

```text
hello
```

要求：

- 只处理白名单群。
- 只响应 @bot。
- 不回复自己。
- 可以 `/help`。
- 可以 `/status`。
- dedup 正常。
- rate limit 正常。

仍然不调用 LLM。

---

# 40. 第三阶段验收标准

Phase 3：

接入 LLM。

实现：

```text
@bot 什么是 Transformer
```

机器人正常回复。

要求：

- OpenAI-compatible API。
- timeout。
- exception handling。
- concurrency limit。
- group lock。
- reply segment。
- 长文本拆分。
- 测试通过。

---

# 41. 第四阶段验收标准

Phase 4：

加入 MemorySessionStore。

验证：

```text
A：
@bot Python和Java有什么区别

Bot：
...

B：
@bot 那性能方面呢？
```

机器人能够理解：

```text
“那性能方面”
```

对应上一轮主题。

同时：

```text
/clear
```

以后再次：

```text
@bot 那性能呢？
```

不应该继续使用清除前上下文。

---

# 42. Codex 工作方式

不要一次实现整个项目。

严格按照：

```text
Phase 1
↓
tests
↓
运行验证
↓
Phase 2
↓
tests
↓
Phase 3
↓
tests
↓
Phase 4
```

每个阶段开始前：

1. 阅读 `SPEC.md`
2. 检查现有代码
3. 给出本阶段修改计划
4. 然后修改

每个阶段完成后：

1. 运行 pytest
2. 运行静态检查（如果已经配置）
3. 汇报修改文件
4. 汇报测试结果
5. 列出人工验证步骤
6. 不主动开始下一阶段

---

# 43. Codex 禁止事项

除非用户明确要求，否则：

不要：

```text
引入 NoneBot
引入 FastAPI
引入 Redis
引入 PostgreSQL
引入 SQLAlchemy
引入 Docker
引入 LangChain
引入 LangGraph
引入 MCP
引入 RAG
开发 Web UI
修改 NapCat 源码
实现 QQ 协议
```

不要把：

```text
QQ账号
API Key
Access Token
群号
```

硬编码到源代码。

不要为了抽象而抽象。

---

# 44. README

README 至少写：

```text
项目作用
架构
安装 uv
创建环境
安装依赖
.env 配置
NapCat 配置
运行命令
测试命令
QQ 中如何测试
常见错误
安全注意事项
```

其中 NapCat 配置明确写：

```text
WebSocket Server
127.0.0.1
3001
messagePostFormat=array
reportSelfMessage=false
token enabled
```

---

# 45. 后续规划

完成 MVP 后才能考虑：

## V2

```text
MemorySessionStore
↓
RedisSessionStore
```

增加：

```text
Redis TTL
跨重启上下文
分布式 rate limit
```

## V3

加入：

```text
PostgreSQL
```

保存：

```text
群配置
模型配置
统计数据
```

## V4

加入：

```text
FastAPI
```

只作为管理 API：

```text
GET  /api/status
GET  /api/groups
PUT  /api/groups/{id}
GET  /api/usage
```

FastAPI 不参与 OneBot 核心消息链路。

## V5

再考虑：

```text
联网搜索
RAG
MCP
Agent
图片理解
语音
群聊总结
```

---

# 46. 当前最重要的原则

第一优先级不是“大模型”。

第一优先级是：

```text
QQ
↓
NapCat
↓
WebSocket
↓
Python
```

这条链路稳定可靠。

第二优先级：

```text
Router
Session
Concurrency
Error Handling
```

最后才是：

```text
LLM
Agent
RAG
MCP
```

保持简单、可测试、可替换。

# 47. 开始工作

首先只执行 Phase 1。

请：

1. 初始化 Python 3.12 + uv 项目。
2. 创建上述最小目录结构。
3. 创建配置系统。
4. 实现 OneBotClient。
5. 实现 WebSocket 自动重连。
6. 实现 echo request-response correlation。
7. 实现最小 Event Dispatcher。
8. 实现 send_group_msg。
9. 编写对应单元测试。
10. 创建 `.env.example`、`.gitignore` 和 README。
11. 运行测试。
12. 停止，不要实现 LLM。

完成后向我汇报：

- 新增/修改了哪些文件。
- OneBotClient 的工作原理。
- 如何配置 NapCat。
- 如何运行。
- 测试结果。
- 我应该如何手工验证 QQ → NapCat → Python → QQ 链路。