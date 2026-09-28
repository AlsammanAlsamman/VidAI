"""MCP server: exposes VidAI tools to Claude (Claude Code: see .mcp.json)."""
from __future__ import annotations

from . import service

INSTRUCTIONS = """VidAI lets you record, understand and edit videos.
Workflow: brief_questions (ask the user) -> studio_start (your anchor config; opens the VidAI Recorder)
-> live_stats / live_control / live_processor while recording (read live_guide first)
-> studio_wait -> or analyze_video (existing file)
-> anchors (summary first, then drill down) -> frame/contact_sheet to look -> plan (build edits)
-> render_video. Times are always source-video seconds. Try regular ops first; train a lab model
(train, one round at a time, adjust until suitable) only when regular ops are not good enough."""

TOOLS = [
    service.brief_questions, service.save_brief, service.suggest_stats,
    service.studio_devices, service.studio_start, service.studio_status, service.studio_wait,
    service.studio_recover,
    service.live_guide, service.live_stats, service.live_control, service.live_processor, service.live_status,
    service.live_wait_request, service.live_effects, service.live_effect,
    service.vidai_install, service.vidai_download, service.vidai_create_file, service.vidai_permissions,
    service.vidai_profile, service.vidai_learn, service.vidai_forget,
    service.live_notify, service.live_ask_user, service.live_say,
    service.model_search, service.model_apply, service.color_from_photo,
    service.studio_sessions,
    service.record_start, service.record_mark, service.record_stop,
    service.analyze_video, service.anchors, service.add_anchor_events,
    service.frame, service.contact_sheet,
    service.plan, service.render_video,
    service.models, service.train, service.classify_video,
]


def build_server():
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("vidai", instructions=INSTRUCTIONS)
    for fn in TOOLS:
        server.add_tool(fn, name=fn.__name__)
    return server


def main() -> None:
    build_server().run("stdio")


if __name__ == "__main__":
    main()
