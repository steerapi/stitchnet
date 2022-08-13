import asyncio
from pyppeteer import launch
import netron
import os

def start_netron(filename):
    netron.start(filename, address=("0.0.0.0",8081))

async def render_graph(filename):
    toPath = os.path.dirname(filename)
    if os.path.exists(f'{filename}.svg'):
        return
    browser = await launch(headless=True)
    page = await browser.newPage()
    await page._client.send("Page.setDownloadBehavior", {
        "behavior": "allow", 
        "downloadPath": f"{toPath}" # TODO set your path
    })
    await page.goto('http://0.0.0.0:8081')
    await page.waitForSelector('#menu-button > svg', {"visible": True, "timeout": 60000})
    await page.click('#menu-button > svg')
    await page.waitForSelector('#menu-dropdown > button:nth-child(16)', {"visible": True, "timeout": 5000})
    await page.click('#menu-dropdown > button:nth-child(16)')
    await page.screenshot({'path': f'{filename}.png'})
    await browser.close()
    return