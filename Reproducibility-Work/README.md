# FTEC5660 Individual Project - Reproducibility Work

## Part 1: Installation and Discord Connection Guide

### 1.1 Installation

**Method 1: Install from source** (recommended for development)

```bash
git clone https://github.com/HKUDS/nanobot.git
cd nanobot
pip install -e .
```

**Method 2: Install with uv** (stable, fast)

```bash
uv tool install nanobot-ai
```

**Method 3: Install from PyPI** (stable)

```bash
pip install nanobot-ai
```

### 1.2 Quick Start

**Step 1: Initialize**

```bash
nanobot onboard
```

**Step 2: Configure API Key** (`~/.nanobot/config.json`)

Add your API key (e.g., OpenRouter):

```json
{
  "providers": {
    "openrouter": {
      "apiKey": "sk-or-v1-xxx"
    }
  }
}
```

Set your model:

```json
{
  "agents": {
    "defaults": {
      "model": "anthropic/claude-opus-4-5"
    }
  }
}
```

**Step 3: Test CLI Chat**

```bash
nanobot agent
```

### 1.3 Discord Connection Setup

**Step 1: Create a Discord Bot**

1. Go to https://discord.com/developers/applications
2. Click "New Application" → Give it a name → Create
3. Navigate to "Bot" in the left sidebar → Click "Add Bot"
4. Click "Reset Token" to generate a new bot token → **Copy and save this token**

**Step 2: Enable Intents**

In the Bot settings:
1. Enable **MESSAGE CONTENT INTENT** (required for reading message content)
2. (Optional) Enable **SERVER MEMBERS INTENT** if you plan to use allow lists based on member data
3. Click "Save Changes"

**Step 3: Get Your User ID**

1. In Discord, go to Settings → Advanced → Enable **Developer Mode**
2. Right-click your avatar in any chat → Click **Copy User ID**

**Step 4: Configure nanobot** (`~/.nanobot/config.json`)

Add Discord channel configuration:

```json
{
  "channels": {
    "discord": {
      "enabled": true,
      "token": "YOUR_BOT_TOKEN",
      "allowFrom": ["YOUR_USER_ID"]
    }
  }
}
```

Replace:
- `YOUR_BOT_TOKEN`: The bot token copied in Step 1
- `YOUR_USER_ID`: Your Discord user ID copied in Step 3

**Step 5: Invite the Bot to Your Server**

1. In Discord Developer Portal, go to OAuth2 → URL Generator
2. Under "Scopes", check `bot`
3. Under "Bot Permissions", check:
   - `Send Messages`
   - `Read Message History`
4. Copy the generated invite URL at the bottom
5. Open the URL in your browser and add the bot to your server

**Step 6: Run the Gateway**

```bash
nanobot gateway
```

### 1.4 Verify Tool Call Visibility in Discord

After starting the gateway, send a message to your bot on Discord that requires tool usage:

```
User: What files are in the workspace?
```

Expected response format:

```
🔧 **list_dir**(path="/home/user/.nanobot/workspace")
✅ **list_dir**: MEMORY.md, HISTORY.md, HEARTBEAT.md

Your workspace contains 3 files:
- MEMORY.md
- HISTORY.md
- HEARTBEAT.md
```

### 1.5 Troubleshooting

| Issue | Solution |
|-------|----------|
| Bot not responding | Check `enabled: true` and correct `token` in config |
| Bot can't read messages | Ensure MESSAGE CONTENT INTENT is enabled |
| Permission denied errors | Check bot has Send Messages permission in the server |
| Only some users can chat | Check `allowFrom` list contains correct user IDs |
| Gateway crashes | Check logs with `nanobot status` or run with `--logs` flag |

### 1.6 CLI Commands Reference

| Command | Description |
|---------|-------------|
| `nanobot onboard` | Initialize config & workspace |
| `nanobot agent -m "..."` | Chat with the agent (single message) |
| `nanobot agent` | Interactive chat mode |
| `nanobot gateway` | Start the gateway (for Discord/Telegram/etc.) |
| `nanobot status` | Show current status |
| `nanobot channels status` | Show channel connection status |

---

## Modification Overview

This part records the details of adding custom tool usage rules to the nanobot Agent framework and implementing tool call visibility in conversations.


## Part 2: Tool Usage Rules (context.py)

### Modified File

**File Path**: `nanobot/agent/context.py`

**Method**: `_get_identity()` (Lines 75-127)

### Original Code (Lines 96-105)

```python
## Tool Call Guidelines
- Before calling tools, you may briefly state your intent (e.g. "Let me check that"), but NEVER predict or describe the expected result before receiving it.
- Before modifying a file, read it first to confirm its current content.
- Do not assume a file or directory exists — use list_dir or read_file to verify.
- After writing or editing a file, re-read it if accuracy matters.
- If a tool call fails, analyze the error before retrying with a different approach.
```

### Added Content

Added two new sections after "Tool Call Guidelines":

#### 1. Tool Usage Rules

```python
## Tool Usage Rules
1. **Necessity Check**: Before using any tool, think whether it is truly needed. Avoid unnecessary tool calls.
2. **Parameter Validation**: Ensure all provided parameters are complete and correct before execution.
3. **Result Analysis**: After tool execution, you MUST analyze the result and provide a human-readable summary.
4. **Safety First**: Avoid using dangerous shell commands that could harm the system or data.
5. **Efficiency Consideration**: Choose the most appropriate tool for the task. Avoid repeated use of the same tool unnecessarily.
6. **Error Handling**: When a tool execution fails, analyze the cause and try alternative approaches.
```

#### 2. Tool Usage Workflow

```python
## Tool Usage Workflow
1. Clarify what information is needed
2. Select the most appropriate tool
3. Explain why this tool is being used
4. Execute the tool call
5. Analyze the execution result
6. Provide summary and next steps

Expected Behavior: The Agent will explain which tool is used at each step, why the tool is used, and each decision will be returned to the user through the conversation.
```

---

## Part 3: Tool Call Visibility in Conversations (loop.py)

### Modified File

**File Path**: `nanobot/agent/loop.py`

### Modification 1: Enhanced `_tool_hint` Method (Lines 157-177)

**Original Code**:
```python
@staticmethod
def _tool_hint(tool_calls: list) -> str:
    """Format tool calls as concise hint, e.g. 'web_search("query")'."""
    def _fmt(tc):
        val = next(iter(tc.arguments.values()), None) if tc.arguments else None
        if not isinstance(val, str):
            return tc.name
        return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
    return ", ".join(_fmt(tc) for tc in tool_calls)
```

**New Code**:
```python
@staticmethod
def _tool_hint(tool_calls: list) -> str:
    """Format tool calls as detailed hint with reasoning."""
    def _fmt(tc):
        args_str = ""
        if tc.arguments:
            parts = []
            for k, v in tc.arguments.items():
                if isinstance(v, str):
                    val = f'"{v[:50]}…"' if len(v) > 50 else f'"{v}"'
                elif isinstance(v, (dict, list)):
                    val = "…" if len(str(v)) > 30 else str(v)
                else:
                    val = str(v)
                parts.append(f"{k}={val}")
            args_str = ", ".join(parts)
        return f"🔧 **{tc.name}**({args_str})"
    return "\n".join(_fmt(tc) for tc in tool_calls)
```

### Modification 2: New `_tool_result_summary` Method (Lines 179-190)

**Added Code**:
```python
@staticmethod
def _tool_result_summary(tool_name: str, result: str, max_len: int = 200) -> str:
    """Format tool result as a concise summary for user visibility."""
    if not result:
        return f"✅ **{tool_name}**: Completed (no output)"
    if len(result) <= max_len:
        return f"✅ **{tool_name}**: {result}"
    return f"✅ **{tool_name}**: {result[:max_len]}…\n_(Result truncated, {len(result)} chars total)_"
```

### Modification 3: Send Tool Result Summary (Lines 243-246)

**Original Code**:
```python
for tool_call in response.tool_calls:
    tools_used.append(tool_call.name)
    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
    result = await self.tools.execute(tool_call.name, tool_call.arguments)
    messages = self.context.add_tool_result(
        messages, tool_call.id, tool_call.name, result
    )
```

**New Code**:
```python
for tool_call in response.tool_calls:
    tools_used.append(tool_call.name)
    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
    result = await self.tools.execute(tool_call.name, tool_call.arguments)
    messages = self.context.add_tool_result(
        messages, tool_call.id, tool_call.name, result
    )
    if on_progress:
        await on_progress(self._tool_result_summary(tool_call.name, result), tool_result=True)
```

### Modification 4: Updated `_bus_progress` Function (Lines 415-422)

**Original Code**:
```python
async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
    meta = dict(msg.metadata or {})
    meta["_progress"] = True
    meta["_tool_hint"] = tool_hint
    await self.bus.publish_outbound(OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
    ))
```

**New Code**:
```python
async def _bus_progress(content: str, *, tool_hint: bool = False, tool_result: bool = False) -> None:
    meta = dict(msg.metadata or {})
    meta["_progress"] = True
    meta["_tool_hint"] = tool_hint
    meta["_tool_result"] = tool_result
    await self.bus.publish_outbound(OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
    ))
```

---

## Modification Purpose

These modifications serve two main purposes:

### 1. Standardize Tool Usage Behavior
- **Transparency**: Agent explains each tool usage decision to users
- **Safety**: Prevents dangerous operations
- **Efficiency**: Reduces redundant and unnecessary tool calls
- **Reliability**: Ensures parameter correctness and proper error handling
- **User Experience**: Provides clear feedback and human-readable summaries

### 2. Tool Call Visibility in Discord Conversations
- **Before Tool Execution**: Shows tool name and all parameters with 🔧 icon
- **After Tool Execution**: Shows result summary with ✅ icon
- **Format**: Markdown bold for tool names, truncated for long results

---

## Expected Agent Behavior in Discord

When a user asks a question that requires tool usage, the conversation will show:

```
User: What's the price of TSLA today?

Agent: 🔧 **web_fetch**(url="...")
✅ **web_fetch**: {"url": "https://stockanalysis.com/stocks/tsla/", "finalUrl": "https://stockanalysis.com/stocks/tsla/", "status": 200, ...}

TSLA Stock Price: $409.38
Change: +$9.55 (+2.39%) ⬆️
Last close: Feb 24, 2026, 4:00 PM EST
```

---

## Related Files

- `nanobot/agent/context.py` - Tool usage rules in system prompt
- `nanobot/agent/loop.py` - Tool call visibility implementation
- `nanobot/channels/discord.py` - Discord channel (receives progress messages)
- `nanobot/bus/events.py` - OutboundMessage structure

---

