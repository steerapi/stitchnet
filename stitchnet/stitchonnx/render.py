import asyncio
from pyppeteer import launch
import netron
def start_netron(filename):
    netron.start(filename, address=("0.0.0.0",8081))

async def render_graph(toPath):
    browser = await launch(headless=True)
    page = await browser.newPage()
    await page._client.send("Page.setDownloadBehavior", {
        "behavior": "allow", 
        "downloadPath": f"{toPath}" # TODO set your path
    })
    await page.goto('http://0.0.0.0:8081')
    # await page.waitFor(10000)
    await page.waitForSelector('#menu-button > svg')
    await page.waitFor(3000)

    # exportbtn = await page.querySelector('#menu-dropdown > button:nth-child(16)')
    await page.click('#menu-button > svg')
    await page.waitFor(1000)
    await page.click('#menu-dropdown > button:nth-child(16)')
    # element = await page.querySelector('h1')
    await page.screenshot({'path': 'example1.png'})
    # svgtext = await svg.getProperty("outerHTML")
    # print the article titles
    # print(dir(svgtext.asElement()))
    # svgvalue = await svgtext.jsonValue()
    await browser.close()
    return