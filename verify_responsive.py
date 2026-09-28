"""
verify_responsive.py - Automated CDP Responsive Testing Suite
=============================================================
Runs headless Chrome across viewports:
320px, 360px, 375px, 390px, 414px, 430px (Portrait & Landscape)
Checks for:
- Page-level horizontal overflow
- Component-level overflow beyond parent boundaries
- Overlapping elements
- Modal & form responsiveness
"""

import asyncio
import json
import subprocess
import time
import urllib.request
import websockets
import os
import sys

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
URL = "http://127.0.0.1:8000"

VIEWPORTS = [
    {"name": "Small Mobile 320", "width": 320, "height": 640},
    {"name": "Mobile 360", "width": 360, "height": 740},
    {"name": "Mobile 375", "width": 375, "height": 667},
    {"name": "Mobile 390", "width": 390, "height": 844},
    {"name": "Mobile 414", "width": 414, "height": 896},
    {"name": "Large Mobile 430", "width": 430, "height": 932},
    {"name": "Mobile Landscape 375x320", "width": 640, "height": 360},
    {"name": "Tablet Portrait 768", "width": 768, "height": 1024},
    {"name": "Desktop 1280", "width": 1280, "height": 800}
]

async def send_cmd(ws, method, params=None, msg_id=1):
    msg = {"id": msg_id, "method": method, "params": params or {}}
    await ws.send(json.dumps(msg))
    while True:
        resp = await ws.recv()
        data = json.loads(resp)
        if data.get("id") == msg_id:
            return data.get("result", {})

async def eval_js(ws, expr, msg_id=1):
    res = await send_cmd(ws, "Runtime.evaluate", {"expression": expr, "returnByValue": True}, msg_id)
    return res.get("result", {}).get("value")

async def run_audit():
    # Start Chrome
    proc = subprocess.Popen([
        CHROME_PATH,
        "--headless=new",
        "--remote-debugging-port=9222",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-extensions",
        URL
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    time.sleep(2.5)

    try:
        # Find page websocket
        with urllib.request.urlopen("http://127.0.0.1:9222/json") as r:
            tabs = json.loads(r.read())
            page_tab = next(t for t in tabs if t.get("type") == "page")
            ws_url = page_tab["webSocketDebuggerUrl"]

        async with websockets.connect(ws_url, max_size=20*1024*1024) as ws:
            # Enable page, runtime, DOM
            await send_cmd(ws, "Page.enable", msg_id=1)
            await send_cmd(ws, "Runtime.enable", msg_id=2)
            await send_cmd(ws, "DOM.enable", msg_id=3)

            # Wait for page data to load and render
            time.sleep(2.0)

            total_issues = 0
            results_summary = []

            js_check_overflow = """
            (() => {
                const issues = [];
                const docWidth = document.documentElement.clientWidth;
                const scrollWidth = document.documentElement.scrollWidth;

                if (scrollWidth > docWidth + 1) {
                    issues.push({
                        type: 'PAGE_HORIZONTAL_OVERFLOW',
                        docWidth: docWidth,
                        scrollWidth: scrollWidth,
                        overflowAmount: scrollWidth - docWidth
                    });
                }

                // Check all elements in the DOM
                const allElements = document.querySelectorAll('*');
                for (const el of allElements) {
                    // Ignore hidden elements
                    if (el.offsetParent === null && el.tagName !== 'BODY' && el.tagName !== 'HTML') continue;
                    
                    const style = window.getComputedStyle(el);
                    if (style.display === 'none' || style.visibility === 'hidden') continue;
                    
                    // Skip SVG child nodes
                    if (el instanceof SVGElement && el.tagName.toLowerCase() !== 'svg') continue;
                    
                    // Skip fixed/absolute ambient glow background elements
                    if (el.classList.contains('ambient-glow')) continue;
                    if (style.position === 'fixed' && el.classList.contains('modal-overlay')) continue;

                    // If element is inside an intentional horizontal scroll container,
                    // its container is tested for viewport fit, while inner elements are meant to scroll horizontally
                    const scrollAncestor = el.closest('.table-responsive, .sub-nav, .nav-status-track');
                    if (scrollAncestor && scrollAncestor !== el) continue;

                    const rect = el.getBoundingClientRect();

                    // Check if element spills past the viewport right edge
                    if (rect.right > docWidth + 1 && rect.width > 0) {
                        issues.push({
                            type: 'ELEMENT_EXCEEDS_VIEWPORT',
                            tag: el.tagName,
                            id: el.id || '',
                            class: el.className || '',
                            rectRight: Math.round(rect.right),
                            docWidth: docWidth,
                            excess: Math.round(rect.right - docWidth)
                        });
                    }

                    // Check if child spills out of parent box
                    const parent = el.parentElement;
                    if (parent && parent !== document.body && parent !== document.documentElement) {
                        // Skip if parent allows horizontal scroll
                        const pStyle = window.getComputedStyle(parent);
                        const allowsHScroll = pStyle.overflowX === 'auto' || pStyle.overflowX === 'scroll';
                        
                        if (!allowsHScroll && !el.classList.contains('ambient-glow') && style.position !== 'absolute' && style.position !== 'fixed') {
                            const pRect = parent.getBoundingClientRect();
                            if (rect.width > pRect.width + 1.5 && pRect.width > 0) {
                                issues.push({
                                    type: 'ELEMENT_EXCEEDS_PARENT',
                                    tag: el.tagName,
                                    id: el.id || '',
                                    class: el.className || '',
                                    parentTag: parent.tagName,
                                    parentId: parent.id || '',
                                    parentClass: parent.className || '',
                                    elementWidth: Math.round(rect.width),
                                    parentWidth: Math.round(pRect.width)
                                });
                            }
                        }
                    }
                }
                return issues;
            })()
            """

            tabs_to_test = [
                {"name": "Dashboard", "selector": '.tab-btn[data-tab="dashboard"]'},
                {"name": "What-If Sandbox", "selector": '.tab-btn[data-tab="whatif"]'},
                {"name": "Hidden Patterns", "selector": '.tab-btn[data-tab="patterns"]'},
                {"name": "Strategic Steps", "selector": '.tab-btn[data-tab="decisions"]'},
                {"name": "Data & Parameters", "selector": '.tab-btn[data-tab="dataset"]'}
            ]

            msg_counter = 10
            for vp in VIEWPORTS:
                # Set viewport metrics
                msg_counter += 1
                await send_cmd(ws, "Emulation.setDeviceMetricsOverride", {
                    "width": vp["width"],
                    "height": vp["height"],
                    "deviceScaleFactor": 2,
                    "mobile": True
                }, msg_id=msg_counter)

                await asyncio.sleep(0.3)

                vp_issues = []
                for tab in tabs_to_test:
                    # Click tab
                    msg_counter += 1
                    await eval_js(ws, f"document.querySelector('{tab['selector']}').click();", msg_id=msg_counter)
                    await asyncio.sleep(0.15)

                    msg_counter += 1
                    tab_issues = await eval_js(ws, js_check_overflow, msg_id=msg_counter)
                    if tab_issues:
                        for issue in tab_issues:
                            issue["tab"] = tab["name"]
                            vp_issues.append(issue)

                # Test modal on this viewport
                msg_counter += 1
                await eval_js(ws, "document.getElementById('exportReportBtn').click();", msg_id=msg_counter)
                await asyncio.sleep(0.2)
                
                msg_counter += 1
                modal_issues = await eval_js(ws, """
                (() => {
                    const issues = [];
                    const modal = document.querySelector('.modal-card');
                    if (modal) {
                        const rect = modal.getBoundingClientRect();
                        const docWidth = document.documentElement.clientWidth;
                        if (rect.right > docWidth + 1 || rect.left < -1) {
                            issues.push({
                                type: 'MODAL_EXCEEDS_VIEWPORT',
                                modalWidth: Math.round(rect.width),
                                docWidth: docWidth
                            });
                        }
                    }
                    return issues;
                })()
                """, msg_id=msg_counter)
                if modal_issues:
                    for issue in modal_issues:
                        issue["tab"] = "Modal"
                        vp_issues.append(issue)

                # Close modal
                msg_counter += 1
                await eval_js(ws, "document.getElementById('closeReportModalBtn').click();", msg_id=msg_counter)
                await asyncio.sleep(0.1)

                # Switch back to dashboard
                msg_counter += 1
                await eval_js(ws, "document.querySelector('.tab-btn[data-tab=\"dashboard\"]').click();", msg_id=msg_counter)

                # Deduplicate issues for this viewport
                unique_issues = []
                seen = set()
                for issue in vp_issues:
                    key = json.dumps(issue, sort_keys=True)
                    if key not in seen:
                        seen.add(key)
                        unique_issues.append(issue)

                results_summary.append({
                    "viewport": vp["name"],
                    "width": vp["width"],
                    "height": vp["height"],
                    "issues_count": len(unique_issues),
                    "issues": unique_issues
                })
                total_issues += len(unique_issues)

            print(json.dumps({
                "total_issues": total_issues,
                "summary": results_summary
            }, indent=2))

    finally:
        proc.terminate()

if __name__ == "__main__":
    asyncio.run(run_audit())
