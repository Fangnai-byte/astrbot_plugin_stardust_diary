#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Memory - AstrBot 智能记忆插件
====================================
- 监听群消息：所有消息先进入短期缓存（保留一天）
- 疑似重要的消息由 LLM 判断，重要则摘要进长期记忆
- 群聊触发 LLM 请求时，检索本群相关长期记忆注入 system_prompt
- 按群隔离：群与群之间的记忆互不可见
"""
import asyncio
import json
import os
import re
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timedelta

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import EventMessageType, PermissionType
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.agent.message import TextPart
from astrbot.core.message.components import Plain
from astrbot.core.platform.message_type import MessageType

try:
    from astrbot.core.utils.astrbot_path import get_astrbot_data_path
except ImportError:  # 兼容旧版本 AstrBot
    def get_astrbot_data_path() -> str:
        return os.path.realpath(os.path.join(os.getcwd(), "data"))


# 命中这些词的消息才值得交给 LLM 判断（节省 token）
class SmartMemory(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        data_dir = get_astrbot_data_path()
        self.db_dir = os.path.join(data_dir, "smart_memory")
        os.makedirs(self.db_dir, exist_ok=True)
        self.db_path = os.path.join(self.db_dir, "memory.db")
        self._init_db()
        self._cleanup_expired()
        self._organizing: set = set()  # 正在整理的群，避免并发重复触发

    # ---------------- 数据库 ----------------
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with closing(self._conn()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    user_name TEXT DEFAULT '',
                    content TEXT NOT NULL,
                    keywords TEXT DEFAULT '',
                    raw TEXT DEFAULT '',
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_long_group ON long_term(group_id)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS short_term (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    user_name TEXT DEFAULT '',
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expire_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_short_group ON short_term(group_id)"
            )

    def _cleanup_expired(self):
        try:
            with closing(self._conn()) as conn, conn:
                conn.execute(
                    "DELETE FROM short_term WHERE expire_at < ?", (time.time(),)
                )
        except Exception as e:
            logger.warning(f"[SmartMemory] 清理过期短期记忆失败: {e}")

    # ---------------- 消息监听 ----------------
    @filter.event_message_type(EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        # 忽略 bot 自己的消息，避免自我循环
        if str(event.get_sender_id()) == str(event.get_self_id()):
            return
        text = event.get_message_str().strip()
        if not text:
            return
        # 插件自身指令不参与记忆
        if text.startswith("/mem"):
            return
        group_id = str(event.get_group_id())
        user_id = str(event.get_sender_id())
        user_name = event.get_sender_name() or user_id

        # 1. 消息进短期缓冲
        self._add_short(group_id, user_id, user_name, text)

        # 2. 满阈值触发 AI 整理（异步，不阻塞回复）
        if self.config.get("organize_enabled", True):
            n = self._count_short(group_id)
            if n >= int(self.config.get("organize_threshold", 100)):
                asyncio.create_task(self._organize(group_id))

    # ---------------- 满阈值 AI 整理 ----------------
    async def _organize(self, group_id: str):
        """攒满阈值后：AI 提炼人物画像键值+要点记忆，存长期后清空缓冲。"""
        if group_id in self._organizing:
            self._log(f"[星尘手账] 群 {group_id} 正在整理中，跳过本次触发")
            return
        self._organizing.add(group_id)
        try:
            self._log(f"[星尘手账] _organize 被调用，群 {group_id}")
            msgs = self._all_short(group_id)
            self._log(f"[星尘手账] 群 {group_id} 缓冲 {len(msgs)} 条")
            if len(msgs) < int(self.config.get("organize_threshold", 100)):
                return
            text = "\n".join(
                f"{r['user_name']}({r['user_id']}): {r['content']}" for r in msgs
            )[:12000]
            provider = self._pick_provider()
            self._log(f"[星尘手账] provider: {provider.meta().id if provider else None}")
            if provider is None:
                return
            resp = await provider.text_chat(
                prompt=text,
                system_prompt=(
                    "你是群聊档案管理员。根据聊天记录做两件事：\n"
                    "1. 为活跃成员建立人物画像，用键值对记录稳定属性（如 生日/喜欢/讨厌/身份/口头禅），"
                    "只记录明确出现过的信息，不要编造。\n"
                    "2. 提取值得长期记住的要点（约定、事件、计划、重要信息）。\n"
                    "只输出一个JSON对象："
                    '{"profiles": [{"user": "昵称", "attrs": {"键": "值"}}], '
                    '"memories": [{"content": "要点", "keywords": ["关键词"]}]}'
                    + (f"\n\n额外要求：{self.config.get('organize_prompt', '')}"
                       if self.config.get("organize_prompt", "") else "")
                ),
            )
            out = "".join(
                [c.text for c in (resp.result_chain.chain if resp.result_chain else []) if isinstance(c, Plain)]
            )
            self._log(f"[星尘手账] LLM 返回前100字: {out[:100]!r}")
            data = self._parse_json(out)
            self._log(f"[星尘手账] 解析结果: {bool(data)}")
            if not data:
                return
            saved = 0
            # 人物画像
            for p in data.get("profiles") or []:
                uname = str(p.get("user", "")).strip() or "未知"
                attrs = p.get("attrs") or {}
                if not attrs:
                    continue
                kv = "；".join(f"{k}={v}" for k, v in attrs.items())
                self._add_long(
                    group_id, "system", uname, f"【画像】{uname}：{kv}",
                    [uname] + list(attrs.keys()), "人物画像",
                )
                saved += 1
            # 要点记忆
            for m in data.get("memories") or []:
                content = str(m.get("content", "")).strip()
                if not content:
                    continue
                self._add_long(
                    group_id, "system", "星尘手账", content[:200],
                    m.get("keywords") or [], "要点提取",
                )
                saved += 1
            # 清空缓冲，重新计数
            self._clear_short(group_id)
            self._log(f"[星尘手账] 群 {group_id} 整理完成，新增 {saved} 条长期记忆")
        except Exception as e:
            self._log(f"[星尘手账] AI 整理失败: {e}")
        finally:
            self._organizing.discard(group_id)

    def _log(self, msg: str):
        """双写日志：astrbot 日志 + 独立文件（防 group_log_archive 清空源日志丢失）"""
        try:
            with open(os.path.join(self.db_dir, "plugin.log"), "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        except Exception:
            pass

    @staticmethod
    def _parse_json(s: str):
        """从 LLM 输出中提取 JSON 对象（容忍 markdown 包裹或前后废话）。"""
        s = (s or "").strip()
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            s = m.group(0)
        try:
            return json.loads(s)
        except Exception:
            return None

    def _pick_provider(self):
        """按配置选择 AI 整理用的供应商；未配置则用当前对话模型。"""
        try:
            pid = str(self.config.get("organize_provider", "") or "")
            if pid:
                p = self.context.get_provider_by_id(pid)
                if p is not None:
                    return p
                logger.warning(f"[星尘手账] 供应商 {pid} 不存在，改用当前模型")
            return self.context.get_using_provider()
        except Exception as e:
            logger.warning(f"[星尘手账] 选择供应商失败: {e}")
            return None

    def _save_plugin_config(self) -> None:
        """把当前 self.config 写回插件配置文件（指令修改配置时用）"""
        try:
            cfg_path = os.path.join(
                get_astrbot_data_path(), "config",
                "astrbot_plugin_stardust_diary_config.json",
            )
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[星尘手账] 保存插件配置失败: {e}")

    def _count_short(self, group_id: str) -> int:
        try:
            with closing(self._conn()) as conn:
                return conn.execute(
                    "SELECT COUNT(*) FROM short_term WHERE group_id = ?",
                    (group_id,),
                ).fetchone()[0]
        except Exception:
            return 0

    def _all_short(self, group_id: str, limit: int = 500):
        try:
            with closing(self._conn()) as conn:
                return conn.execute(
                    "SELECT * FROM short_term WHERE group_id = ? "
                    "ORDER BY created_at ASC LIMIT ?",
                    (group_id, limit),
                ).fetchall()
        except Exception:
            return []

    def _clear_short(self, group_id: str):
        try:
            with closing(self._conn()) as conn, conn:
                conn.execute(
                    "DELETE FROM short_term WHERE group_id = ?", (group_id,)
                )
        except Exception as e:
            logger.warning(f"[星尘手账] 清空缓冲失败: {e}")

    # ---------------- 存储 ----------------
    def _add_short(self, group_id, user_id, user_name, text):
        now = time.time()
        try:
            with closing(self._conn()) as conn, conn:
                conn.execute(
                    "INSERT INTO short_term (group_id, user_id, user_name, content, created_at, expire_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (group_id, user_id, user_name, text[:500], now, now + 86400),
                )
        except Exception as e:
            logger.warning(f"[SmartMemory] 短期记忆写入失败: {e}")

    def _add_long(self, group_id, user_id, user_name, summary, keywords, raw):
        try:
            with closing(self._conn()) as conn, conn:
                conn.execute(
                    "INSERT INTO long_term (group_id, user_id, user_name, content, keywords, raw, created_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (
                        group_id,
                        user_id,
                        user_name,
                        summary,
                        ",".join(keywords[:8]),
                        raw[:500],
                        time.time(),
                    ),
                )
        except Exception as e:
            logger.warning(f"[SmartMemory] 长期记忆写入失败: {e}")

    # ---------------- 检索 ----------------
    def _search_long(self, group_id: str, query: str, top_k: int = 5):
        query = (query or "").strip()
        try:
            with closing(self._conn()) as conn:
                rows = conn.execute(
                    "SELECT * FROM long_term WHERE group_id = ? ORDER BY created_at DESC LIMIT 300",
                    (group_id,),
                ).fetchall()
        except Exception as e:
            logger.warning(f"[SmartMemory] 检索失败: {e}")
            return []
        scored = []
        for r in rows:
            score = 0
            for kw in (r["keywords"] or "").split(","):
                if kw and kw in query:
                    score += 2
            content = r["content"] or ""
            if query and query in content:
                score += 3
            for gram in self._grams(query):
                if gram and gram in content:
                    score += 1
            if score > 0:
                scored.append((score, r))
        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:top_k]]

    @staticmethod
    def _grams(s: str, n: int = 4):
        s = re.sub(r"\s+", "", s or "")
        return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}

    def _since_ts(self, level: str) -> float:
        """分层时间起点：L1长期全部 / L2今天 / L3最近三天 / L4本周一。"""
        now = datetime.now()
        if level == "L2":
            return datetime(now.year, now.month, now.day).timestamp()
        if level == "L3":
            return (now - timedelta(days=3)).timestamp()
        if level == "L4":
            week_start = (now - timedelta(days=now.weekday())).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            return week_start.timestamp()
        return 0.0  # L1：全部长期记忆

    def _detect_time_intent(self, query: str) -> str | None:
        """根据提问内容判断时间层级意图。"""
        if any(k in query for k in ("今天", "刚才", "今早", "今晚", "早上", "下午", "中午", "晚上")):
            return "L2"
        if any(k in query for k in ("最近", "这两天", "这几天", "前两天", "前天", "昨天")):
            return "L3"
        if any(k in query for k in ("本周", "这周", "这星期", "这个星期", "星期", "一周")):
            return "L4"
        return None

    def _recent_short_since(self, group_id: str, since_ts: float, n: int = 10):
        try:
            with closing(self._conn()) as conn:
                return conn.execute(
                    "SELECT * FROM short_term WHERE group_id = ? AND created_at >= ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (group_id, since_ts, n),
                ).fetchall()
        except Exception:
            return []

    def _recent_short(self, group_id: str, n: int = 10):
        try:
            with closing(self._conn()) as conn:
                return conn.execute(
                    "SELECT * FROM short_term WHERE group_id = ? AND expire_at > ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (group_id, time.time(), n),
                ).fetchall()
        except Exception:
            return []

    # ---------------- 注入上下文 ----------------
    @filter.on_llm_request()
    async def on_llm_req(self, event: AstrMessageEvent, req: ProviderRequest):
        try:
            if event.get_message_type() != MessageType.GROUP_MESSAGE:
                return
            group_id = str(event.get_group_id())
            query = event.get_message_str().strip()
            if not query:
                return
            parts = []
            # 相关长期记忆
            top_k = int(self.config.get("top_k", 5))
            mems = self._search_long(group_id, query, top_k)
            if mems:
                lines = []
                for r in mems:
                    when = datetime.fromtimestamp(r["created_at"]).strftime("%m-%d")
                    lines.append(f"- ({when} {r['user_name']}) {r['content']}")
                parts.append("【记忆片段，来自本群历史消息，可参考但勿编造】\n" + "\n".join(lines))
            # 分层短期记忆：识别提问的时间意图（L2今天/L3最近几天/L4本周）
            level = self._detect_time_intent(query)
            if level:
                since = self._since_ts(level)
                n = int(self.config.get("recent_count", 5)) * 2
                rows = self._recent_short_since(group_id, since, n)
                label = {"L2": "今天", "L3": "最近几天", "L4": "本周"}.get(level, level)
                if rows:
                    lines = [
                        f"- {r['user_name']}: {r['content']}" for r in reversed(rows)
                    ]
                    parts.append(f"【{label}的群聊记录，用户可能问的是这段时间的事】\n" + "\n".join(lines))
                # 关键词追问引导
                parts.append(
                    "【检索提示】若用户询问时间段内发生的事情但描述模糊，"
                    "请先引导用户说出更具体的关键词（人名/话题/事件），"
                    "再根据关键词检索上面的记忆和【记忆片段】后回答，不要凭空编造。"
                )
            elif self.config.get("include_recent", True):
                n = int(self.config.get("recent_count", 5))
                recent = self._recent_short(group_id, n)
                if recent:
                    lines = [
                        f"- {r['user_name']}: {r['content']}" for r in reversed(recent)
                    ]
                    parts.append("【最近消息，可能与本条相关】\n" + "\n".join(lines))
            if parts:
                # 注入到用户消息之后而非 system_prompt，保持前缀稳定，缓存才能命中
                req.extra_user_content_parts.append(
                    TextPart(text="\n\n".join(parts))
                )
        except Exception as e:
            logger.warning(f"[SmartMemory] 注入记忆失败: {e}")

    # ---------------- 指令 ----------------
    @filter.command("mem", alias={"memory"})
    async def mem(self, event: AstrMessageEvent):
        args = (event.message_str or "").split()
        sub = args[1].strip().lower() if len(args) > 1 else "help"
        group_id = str(event.get_group_id())
        user_id = str(event.get_sender_id())

        if sub == "list":
            n = int(args[2]) if len(args) > 2 and args[2].isdigit() else 10
            try:
                with closing(self._conn()) as conn:
                    rows = conn.execute(
                        "SELECT * FROM long_term WHERE group_id = ? "
                        "ORDER BY created_at DESC LIMIT ?",
                        (group_id, n),
                    ).fetchall()
            except Exception as e:
                yield event.plain_result(f"查询失败: {e}")
                return
            if not rows:
                yield event.plain_result("本群还没有长期记忆哦～")
                return
            lines = [f"本群最近 {len(rows)} 条长期记忆："]
            for r in rows:
                when = datetime.fromtimestamp(r["created_at"]).strftime("%m-%d %H:%M")
                lines.append(f"[{r['id']}] ({when} {r['user_name']}) {r['content']}")
            yield event.plain_result("\n".join(lines))
            return

        if sub == "recent":
            n = int(args[2]) if len(args) > 2 and args[2].isdigit() else 10
            recent = self._recent_short(group_id, n)
            if not recent:
                yield event.plain_result("本群最近没有短期消息记录～")
                return
            lines = [f"本群最近 {len(recent)} 条消息（一天内有效）："]
            for r in reversed(recent):
                when = datetime.fromtimestamp(r["created_at"]).strftime("%H:%M")
                lines.append(f"({when} {r['user_name']}) {r['content']}")
            yield event.plain_result("\n".join(lines))
            return

        if sub == "forget":
            if len(args) < 3 or not args[2].isdigit():
                yield event.plain_result("用法：/mem forget <id>")
                return
            mid = int(args[2])
            try:
                with closing(self._conn()) as conn:
                    row = conn.execute(
                        "SELECT * FROM long_term WHERE id = ? AND group_id = ?",
                        (mid, group_id),
                    ).fetchone()
                    if not row:
                        yield event.plain_result("没有找到这条记忆（只能操作本群的）")
                        return
                    if str(row["user_id"]) != user_id and not event.is_admin():
                        yield event.plain_result("这不是你的记忆，只有管理员能删哦～")
                        return
                    conn.execute("DELETE FROM long_term WHERE id = ?", (mid,))
                yield event.plain_result(f"已删除记忆 [{mid}]：{row['content']}")
            except Exception as e:
                yield event.plain_result(f"删除失败: {e}")
            return

        if sub == "models":
            providers = self.context.get_all_providers()
            if not providers:
                yield event.plain_result("当前没有任何模型供应商哦～")
                return
            lines = ["可用模型供应商："]
            for p in providers:
                m = p.meta
                lines.append(f"- {m.id}（{m.type}）当前模型：{m.model}")
                try:
                    models = await p.get_models()
                    if models:
                        shown = "、".join(models[:10])
                        lines.append(f"  可选：{shown}")
                except Exception:
                    pass
            lines.append("用 /mem model <供应商id> 设置整理用提供商")
            yield event.plain_result("\n".join(lines))
            return

        if sub == "model":
            if len(args) < 3:
                yield event.plain_result("用法：/mem model <供应商id>，先 /mem models 查看")
                return
            pid = args[2]
            self.config["organize_provider"] = pid
            self._save_plugin_config()
            yield event.plain_result(f"AI整理提供商已设为：{pid}")
            return

        if sub == "stat":
            try:
                with closing(self._conn()) as conn:
                    long_n = conn.execute(
                        "SELECT COUNT(*) FROM long_term WHERE group_id = ?", (group_id,)
                    ).fetchone()[0]
                    short_n = conn.execute(
                        "SELECT COUNT(*) FROM short_term WHERE group_id = ? AND expire_at > ?",
                        (group_id, time.time()),
                    ).fetchone()[0]
                yield event.plain_result(
                    f"本群记忆统计：长期 {long_n} 条，短期（一天内）{short_n} 条"
                )
            except Exception as e:
                yield event.plain_result(f"统计失败: {e}")
            return

        yield event.plain_result(
            "Smart Memory 指令：\n"
            "/mem list [n] - 查看长期记忆\n"
            "/mem recent [n] - 查看最近消息\n"
            "/mem forget <id> - 删除记忆\n"
            "/mem stat - 统计\n"
            "/mem help - 帮助"
        )
